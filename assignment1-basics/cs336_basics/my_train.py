import argparse
import os
import random
import time
import numpy as np
import torch
import json
import yaml
from pathlib import Path
from dataclasses import asdict
from cs336_basics import my_nn_models, my_optim, my_data, my_nn_functions, tokenizer
from cs336_basics.config import get_config


def set_all_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


def get_default_config(config_path: str = "config_tinystories.yaml"):
    if os.path.isdir(config_path):
        config = get_config(os.path.join(config_path, "config.yaml"))
        run_dir = Path(config_path)
    else:
        config = get_config(config_path)
        run_dir = Path(config.train.output_dir) / f"{config.train.project}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
    return config, run_dir


def main(config, run_dir):
    device = "cuda" if config.model.device == "cuda" and torch.cuda.is_available() else "cpu"
    my_dtype = get_dtype(config.model.dtype)

    train_dataset = np.memmap(config.data.train_data, dtype=np.uint16, mode="r")
    val_dataset = np.memmap(config.data.val_data, dtype=np.uint16, mode="r")

    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_file = checkpoint_dir / "checkpoint.pt"
    best_checkpoint_file = checkpoint_dir / "best_checkpoint.pt"
    best_val = float("inf")

    set_all_seed(config.train.seed)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    run_config = run_dir / "config.yaml"
    if not run_config.exists():
        with open(run_config, "w") as f:
            yaml.dump(asdict(config), f, indent=2)

    model = my_nn_models.TransformerLM(
        config.model.vocab_size,
        config.model.context_length,
        config.model.d_model,
        config.model.num_layers,
        config.model.num_heads,
        d_ff=config.model.d_ff,
        rope_theta=config.model.rope_theta,
        device=device,
        dtype=my_dtype,
    )
    # model = torch.compile(model)
    optimizer = my_optim.AdamW(
        model.parameters(),
        lr=config.scheduler.max_learning_rate,
        weight_decay=config.optim.weight_decay,
        betas=config.optim.betas,
        eps=config.optim.eps,
    )

    start = 0
    if config.train.wandb.enabled:
        import wandb

        wandb.init(project=config.train.wandb.project, name=config.train.wandb.name, config=config)

    if os.path.exists(checkpoint_file):
        start = my_data.run_load_checkpoint(checkpoint_file, model, optimizer)
        print(f"Loading checkpoint from {checkpoint_file}, starting from step {start}")

    # torch.autograd.set_detect_anomaly(True)
    for t in range(start, config.train.max_steps):
        lr = my_optim.learning_rate_schedule(
            t,
            config.scheduler.max_learning_rate,
            config.scheduler.min_learning_rate,
            config.scheduler.warmup_iters,
            config.scheduler.cosine_cycle_iters,
        )
        my_optim.set_learning_rate(optimizer, lr)
        x, y = my_data.data_loading(train_dataset, config.train.batch_size, config.model.context_length, device)
        logits = model(x)
        loss = my_nn_functions.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        my_optim.gradient_clipping(model.parameters(), config.train.max_l2_norm)
        optimizer.step()

        if (t + 1) % config.train.log_interval == 0:
            print(f"Step {t + 1}: loss = {loss.item():.4f}, lr = {lr:.2e}")
            if config.train.wandb.enabled:
                wandb.log({"loss": loss.item(), "lr": lr, "iteration": t + 1})

        if (t + 1) % config.train.eval_interval == 0:
            val_loss = 0.0
            for _ in range(10):
                x, y = my_data.data_loading(val_dataset, config.train.batch_size, config.model.context_length, device)
                logits = model(x)
                val_loss += my_nn_functions.cross_entropy(logits, y).item()
            val_loss /= 10
            print(f"Step {t + 1}: val_loss = {val_loss:.4f}")
            if config.train.wandb.enabled:
                wandb.log({"val_loss": val_loss, "lr": lr, "iteration": t + 1})
            if val_loss < best_val:
                best_val = val_loss
                my_data.save_checkpoint(model, optimizer, t + 1, best_checkpoint_file)

        if (t + 1) % config.train.checkpoint_interval == 0:
            my_data.save_checkpoint(model, optimizer, t + 1, checkpoint_file)
    my_data.save_checkpoint(model, optimizer, t + 1, checkpoint_file)

    tok = tokenizer.Tokenizer.from_files(config.data.vocab_path, config.data.merge_path)
    # generate
    prompt = "Once upon a time"
    prompt_ids = tok.encode(torch.tensor(prompt_ids))
    answer_ids = model.generate(prompt_ids, max_new_tokens=100, eot_token_id=tok.encode("<|endoftext|>"))
    print(f"with prompt: {prompt}\n:{tok.decode(answer_ids)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_tinystories.yaml", help="Path to config file")
    args = parser.parse_args()
    config, run_dir = get_default_config(args.config)
    main(config, run_dir)
