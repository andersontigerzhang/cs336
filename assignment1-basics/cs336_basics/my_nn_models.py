import torch
import math
from torch import Tensor
from jaxtyping import Float, Int, Bool
from einops import rearrange, repeat
from cs336_basics import utils, my_nn_functions

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, 
                 out_features: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        weight_std = math.sqrt(2.0 / (in_features+out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=weight_std, a=-3*weight_std, b=3*weight_std)

    def forward(self, 
                x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        
        x = torch.einsum('oi, ... i -> ... o', self.weight, x)
        return x

class Embedding(torch.nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, x: Int[Tensor, "..."]) -> Float[Tensor, "... d_emb"]:
        return self.weight[x]
    
class RMSNorm(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 *,
                 eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        torch.nn.init.ones_(self.weight)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x1 = x.to(torch.float32)  # to prevent from overflow when computing the norm
        rms_x = torch.einsum('... d -> ...', x1**2)
        rms_x = torch.sqrt(rms_x/self.d_model + self.eps)
        return (x1 / rms_x.unsqueeze(-1) * self.weight).to(dtype=x.dtype)
        
class SwiGLU(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        
        super().__init__()
        if d_ff is None:
            d_ff = int(64 * (d_model * 8/3 // 64))
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        gate = self.w1(x)
        data = self.w3(x)
        return self.w2(my_nn_functions.silu(gate) * data)

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None = None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {d_k}.")
        
        ks = torch.arange(start=0, end=d_k, step=2,dtype=torch.float32, device=device)  # 2k-2
        freq = 1 / (theta ** (ks/d_k))
        pos = torch.arange(end=max_seq_len, dtype=torch.float32, device=device)
        theta_i_k = torch.einsum('i, k -> i k', pos, freq)  # (max_seq_len, d_k//2)
        cos_theta = torch.cos(theta_i_k)
        sin_theta = torch.sin(theta_i_k)
        self.register_buffer("cos_theta", cos_theta, persistent=False)
        self.register_buffer("sin_theta", sin_theta, persistent=False)

    # %% [markdown]
    # '''
    # If our original vector is $\mathbf{x} = [x_1, x_2, x_3, x_4, \dots, x_d]$, 
    # then we define $\mathbf{\tilde{x}}$ as:$$\mathbf{\tilde{x}} = [-x_2, x_1, -x_4, x_3, \dots, -x_d, x_{d-1}]$$

    # $$\text{RoPE}(\mathbf{x}, m) = \mathbf{x} \odot \cos(m\theta) + \mathbf{\tilde{x}} \odot \sin(m\theta)$$
    # '''
    # %%
    def rotate_every_n(self, x: Float[Tensor, "... seq_len d_k"],
                       n: int | None = 2) -> Float[Tensor, "... seq_len d_k"]:
        x = rearrange(x, '... (d k) -> ... d k', k=n)
        return rearrange([-x[...,1],x[...,0]], 'n ... d -> ...  (d n)')

    @torch.no_grad()
    def forward(self, x: Float[Tensor, "... seq_len d_k"],
                token_positions: Int[Tensor, "seq_len"]) -> Float[Tensor, "... seq_len d_k"]:

        token_positions = token_positions.to(x.device)  # (... seq_len d_k)

        cos_theta = repeat(self.cos_theta[token_positions], '... d -> ... (d 2)')
        sin_theta = repeat(self.sin_theta[token_positions], '... d -> ... (d 2)')
        hadamard = '... ij, ... ij -> ... ij'
        x_fp32 = x.to(torch.float32)  # to prevent from overflow when computing the RoPE
        rotate_x = self.rotate_every_n(x_fp32)

        hadamard = '... ij, ... ij -> ... ij'
        out = torch.einsum(hadamard, cos_theta, x_fp32) + torch.einsum(hadamard, sin_theta, rotate_x)
        return out.to(dtype=x.dtype)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta,
                                                d_k=d_model//num_heads,
                                                max_seq_len=max_seq_len,
                                                device=device)
        else:
            self.rope = None

    def forward(self, x: Float[Tensor, "... seq_len d_model"],
                token_positions: Int[Tensor, "seq_len"] | None = None) -> Float[Tensor, "... seq_len d_model"]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        reshape_rotate = '... seq_len (h d_k) -> ... h seq_len d_k'
        q = rearrange(q, reshape_rotate, h = self.num_heads)
        k = rearrange(k, reshape_rotate, h = self.num_heads)
        v = rearrange(v, reshape_rotate, h = self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                pos = torch.arange(x.shape[-2], device=x.device)
            else:
                pos = token_positions.to(device=x.device)
            q = self.rope(q, token_positions=pos)
            k = self.rope(k, token_positions=pos)

        *batch_dim, seq_len, d_model = q.size()
        mask = torch.tril(torch.ones(seq_len,seq_len, device=x.device, dtype=torch.bool))
        mask = mask.__getitem__((None,) * len(batch_dim) + (...,))
        attn = my_nn_functions.scaled_dot_product_attention(q,k,v,mask=mask)
        attn = rearrange(attn, '... h seq_len d_k -> ... seq_len (h d_k)')
        return self.output_proj(attn)
    
class TransformerBlock(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len,
                                       device=device, dtype=dtype)
        
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self,
                x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        layer1 = self.ln1(x)
        layer1 = self.attn(layer1)
        x = x + layer1
        layer2 = self.ln2(x)
        layer2 = self.ffn(layer2)
        return x + layer2

class TransformerLM(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 *,
                 d_ff: int | None = None,
                 rope_theta: float | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.Sequential()
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(d_model, 
                                        num_heads, 
                                        d_ff=d_ff, 
                                        max_seq_len=context_length, 
                                        theta=rope_theta, 
                                        device=device, 
                                        dtype=dtype))
        self.num_layers = num_layers
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[Tensor, "... batch_size seq_len"]) -> Float[Tensor, "... batch_size seq_len vocab_size"]:
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

    @torch.no_grad()
    def generate(self, x: Int[Tensor, "... batch_size seq_len"], 
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: int | None = None,
                 eot_token_id: int = 256) -> Int[Tensor, "... batch_size seq_len max_new_tokens"]:
        if x.dim() == 1:
            x = x.unsqueeze_(0)
        x_len = x.size(-1)
        x = x.to(device=self.device)
        for _ in range(max_new_tokens):
            x = x[:, -self.context_length:] if x.shape[-1] > self.context_length else x
            logits = self.forward(x)
            next_token_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                topk_values, topk_indices = torch.topk(next_token_logits, k=top_k)
                threashold = topk_values[:, -1]
                topk_mask = next_token_logits < threashold
                next_token_logits.masked_fill_(topk_mask, float('-inf'))
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == eot_token_id:
                break
            x = torch.cat((x, next_token), dim=-1)
        return x[:, x_len:]