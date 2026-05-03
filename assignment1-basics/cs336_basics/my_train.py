import os
import random
import time
import numpy as np
import torch
# import docopt
import yaml

from cs336_basics import my_nn_models, my_optim, my_data

def set_all_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # run
    seed = 42
    project = 'cs336_hw1'
    output_dir = 'output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    my_dtype = torch.float32
    # model
    vocab_size = 10000
    context_length = 1024
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 512
    rope_theta = 10000.
    
    # optim
    weight_decay = 0.01
    betas = (0.9, 0.95)
    eps = 1e-8
    
    # learning rate scheduler
    max_learning_rate = 6e-4
    min_learning_rate = 6e-5
    warmup_iters = 2000
    cosine_cycle_iters = 10000
    
    # train
    max_steps = 10000
    data_path = ""    

    run_dir = os.path.join(output_dir, f"{project}_{time.strftime('%Y%m%d_%H%M%S')}")
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pt')
    best_checkpoint_file = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
    best_val = float('-inf')
    
    os.makedirs(run_dir, exist_ok=True)
    set_all_seed(seed)

    model = my_nn_models.TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=my_dtype
    )

    optimizer = my_optim.AdamW(
        model.parameters(),
        lr=max_learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps)

    start = 0

    if os.path.exists(checkpoint_file):
        start = my_data.run_load_checkpoint(checkpoint_file, model, optimizer)

    for t in range(start, max_steps):
        lr = my_optim.learning_rate_schedule(t, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
        my_optim.set_learning_rate(optimizer, lr)
        x, y = my_data.data_loading(dataset, batch_size, context_length, device)
        logits = model(x)  # batch, seq_len, vocab_size
        loss = my_nn_functions.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        my_optim.gradient_clipping(model.parameters(), max_l2_norm)
        optimizer.step()

if __name__ == '__main__':
    main()