import torch,math
from jaxtyping import Float, Int, Bool
from torch import Tensor
from einops import reduce

def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return x * torch.sigmoid(x)


# %% [markdown]
# Note that exp(𝑣𝑖) can become inf for large values (then, inf/inf=NaN). 
# We can avoid this by noticing that the softmax operation is invariant to adding any constant 𝑐 to all inputs. 
# We can leverage this property for numerical stability—typically, we will subtract the largest entry of 𝑣 
# from all elements of 𝑣, making the new largest entry 0.
# %%  
def softmax(x: Float[Tensor, "..."],
            dim: int = -1) -> Float[Tensor, "..."]:
    x_max = torch.amax(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

def scaled_dot_product_attention(q: Float[Tensor, "batch_size ... seq_len d_k"],
                                 k: Float[Tensor, "batch_size ... seq_len d_k"],
                                 v: Float[Tensor, "batch_size ... seq_len d_v"],
                                 mask: Bool[Tensor, "batch_size ... seq_len seq_len"] | None = None
                                ) -> Float[Tensor, "batch_size ... seq_len d_v"]:

    d_k = q.shape[-1]
    q,k = q.to(torch.float32), k.to(torch.float32)
    logits = torch.einsum('...nd,...md->...nm',q,k)/math.sqrt(d_k)
    if mask is not None:
        logits.masked_fill_(~mask, torch.finfo(v.dtype).min)
    probs = softmax(logits, dim=-1)
    attn = torch.einsum('...nm,...md->...nd',probs,v.to(torch.float32))
    return attn.to(dtype=v.dtype)

def cross_entropy(logits: Float[Tensor, "batch_size ... vocab_size"], 
                  targets: Int[Tensor, "batch_size ... vocab_size"]) -> Float[Tensor, ""]:
    logits -= reduce(logits, "batch_size ... vocab_size -> batch_size ... 1", "max")
    denominator = reduce(torch.exp(logits), "batch_size ... vocab_size -> batch_size ... 1", "sum")
    rez =  torch.log(denominator) - torch.gather(logits, -1, targets[:,None])
    return torch.mean(rez)