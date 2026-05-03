from cs336_basics import my_nn_models


'''
    a TransformerLM has the following components:
    - token_embeddings: vocab_size x d_model
    - layers(TransformerBlock): num_layers x (d_model + 4 x d_model x d_model + d_model + 3 x d_model x d_ff)
      - TransformerBlock: d_model + 4 x d_model x d_model + d_model + 3 x d_model x d_ff
        - ln1(RMSNorm): d_model
        - attn(MultiHeadAttention): 4 x d_model x d_model
          - output_proj(Linear): d_model x d_model
          - q_proj(Linear): d_model x d_model
          - k_proj(Linear): d_model x d_model
          - v_proj(Linear): d_model x d_model
          - rope(RotaryPositionalEmbedding): 0
        - ln2(RMSNorm): d_model
        - ffn(SwiGLU): 3 x d_model x d_ff
          - w1: d_model x d_ff
          - w2: d_ff x d_model
          - w3: d_model x d_ff
    - ln_final(RMSNorm): d_model
    - lm_head(Linear): vocab_size x d_model

    - total parameters: 1_640_452_800
    - total memory: 1640452800*4 = ~6GB
'''

'''
    # flops calculation:
    - token_embeddings: 0
    - layers(TransformerBlock): num_layers x (d_model + 4 x d_model x d_model + d_model + 3 x d_model x d_ff)
      - TransformerBlock: d_model + 4 x d_model x d_model + d_model + 3 x d_model x d_ff
        - ln1(RMSNorm): d_model
        - attn(MultiHeadAttention): 4 x d_model x d_model
          - output_proj(Linear): d_model x d_model
          - q_proj(Linear): d_model x d_model
          - k_proj(Linear): d_model x d_model
          - v_proj(Linear): d_model x d_model
          - rope(RotaryPositionalEmbedding): 0
        - ln2(RMSNorm): d_model
        - ffn(SwiGLU): 3 x d_model x d_ff
          - w1: d_model x d_ff
          - w2: d_ff x d_model
          - w3: d_model x d_ff
    - ln_final(RMSNorm): d_model
    - lm_head(Linear): vocab_size x d_model

    - total parameters: 1_640_452_800
    - total memory: 1640452800*4 = ~6GB
'''
def print_trainable_params(module, indent=0, recursive=False):
    prefix = " " * (indent * 4)
    
    for name, child in module.named_children():
        child_total = sum(p.numel() for p in child.parameters() if p.requires_grad)
        child_name = child.__class__.__name__
        
        if isinstance(child, (nn.Sequential, nn.ModuleList)):
            length = len(child)
            print(f"{prefix}    {name} ({child_name} x{length}): {child_total:,} trainable params")
            for i, subchild in enumerate(child):
                sub_total = sum(p.numel() for p in subchild.parameters() if p.requires_grad)
                sub_name = subchild.__class__.__name__
                print(f"{prefix}        [{i}] ({sub_name}): {sub_total:,} trainable params")
                if recursive:
                    print_trainable_params(subchild, indent + 2, recursive=True)
        else:
            print(f"{prefix}    {name} ({child_name}): {child_total:,} trainable params")
            if recursive:
                print_trainable_params(child, indent + 1, recursive=True)


def summary(module, recursive=False):
    total = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"{module.__class__.__name__}: {total:,} trainable params")
    print_trainable_params(module, recursive=recursive)

transformlm = my_nn_models.TransformerLM(vocab_size=50_257,
                                         context_length=1_024,
                                         num_layers=48,
                                         d_model=1600,
                                         num_heads=25,
                                         d_ff=4_288,
                                         )

summary(transformlm)