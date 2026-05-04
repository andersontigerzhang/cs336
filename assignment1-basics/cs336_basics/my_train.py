import argparse
import os
import random
import time
import numpy as np
import torch
import yaml
import json

from cs336_basics import my_nn_models, my_optim, my_data, my_nn_functions


def set_all_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


def main(config_path: str = "config_tinystories.yaml"):
    config = load_config(config_path)

    cfg_model = config["model"]
    cfg_optim = config["optim"]
    cfg_scheduler = config["scheduler"]
    cfg_data = config["data"]
    cfg_train = config["train"]

    seed = cfg_train["seed"]
    project = cfg_train["project"]
    output_dir = cfg_train["output_dir"]
    run_dir = config_path if os.path.isdir(config_path) else None

    device_str = cfg_model["device"]
    device = "cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu"
    my_dtype = get_dtype(cfg_model["dtype"])

    vocab_size = cfg_model["vocab_size"]
    context_length = cfg_model["context_length"]
    d_model = cfg_model["d_model"]
    num_layers = cfg_model["num_layers"]
    num_heads = cfg_model["num_heads"]
    d_ff = cfg_model["d_ff"]
    rope_theta = cfg_model["rope_theta"]

    weight_decay = cfg_optim["weight_decay"]
    betas = tuple(cfg_optim["betas"])
    eps = cfg_optim["eps"]

    max_learning_rate = float(cfg_scheduler["max_learning_rate"])
    min_learning_rate = float(cfg_scheduler["min_learning_rate"])
    warmup_iters = cfg_scheduler["warmup_iters"]
    cosine_cycle_iters = cfg_scheduler["cosine_cycle_iters"]

    max_steps = cfg_train["max_steps"]
    batch_size = cfg_train["batch_size"]
    max_l2_norm = cfg_train["max_l2_norm"]
    log_interval = cfg_train["log_interval"]
    val_interval = cfg_train["val_interval"]
    checkpoint_interval = cfg_train["checkpoint_interval"]

    train_data_path = cfg_data["train_data"]
    val_data_path = cfg_data["val_data"]

    train_dataset = np.memmap(train_data_path, dtype=np.uint16, mode="r")
    val_dataset = np.memmap(val_data_path, dtype=np.uint16, mode="r")

    if run_dir is None:
        run_dir = os.path.join(output_dir, f"{project}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        yaml.dump(config, open(os.path.join(run_dir, "config.yaml"), "w"))

    checkpoint_dir = os.path.join(run_dir, "checkpoint")
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.pt")
    best_checkpoint_file = os.path.join(checkpoint_dir, "best_checkpoint.pt")
    best_val = float("inf")

    set_all_seed(seed)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    model = my_nn_models.TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=my_dtype,
    )

    optimizer = my_optim.AdamW(
        model.parameters(), lr=max_learning_rate, weight_decay=weight_decay, betas=betas, eps=eps
    )

    start = 0

    if os.path.exists(checkpoint_file):
        start = my_data.run_load_checkpoint(checkpoint_file, model, optimizer)

    torch.autograd.set_detect_anomaly(True)
    for t in range(start, max_steps):
        lr = my_optim.learning_rate_schedule(t, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
        my_optim.set_learning_rate(optimizer, lr)
        x, y = my_data.data_loading(train_dataset, batch_size, context_length, device)
        logits = model(x)
        loss = my_nn_functions.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        my_optim.gradient_clipping(model.parameters(), max_l2_norm)
        optimizer.step()

        if (t + 1) % log_interval == 0:
            print(f"Step {t + 1}: loss = {loss.item():.4f}, lr = {lr:.2e}")

        if (t + 1) % val_interval == 0:
            val_loss = 0.0
            for _ in range(10):
                x, y = my_data.data_loading(val_dataset, batch_size, context_length, device)
                logits = model(x)
                val_loss += my_nn_functions.cross_entropy(logits, y).item()
            val_loss /= 10
            print(f"Step {t + 1}: val_loss = {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                my_data.save_checkpoint(model, optimizer, t + 1, best_checkpoint_file)

        if (t + 1) % checkpoint_interval == 0:
            my_data.save_checkpoint(model, optimizer, t + 1, checkpoint_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_tinystories.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
