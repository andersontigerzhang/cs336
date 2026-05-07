import argparse
import sys
from cs336_basics import my_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["lr", "batch"])
    parser.add_argument("values", nargs="+", type=float)
    args = parser.parse_args()

    config = my_train.get_config()

    if args.command == "lr":
        for val in args.values:
            config.train.wandb.name = f"sweep_{args.command}_{val}"
            config.train.wandb.enabled = True
            config.scheduler.max_learning_rate = val
            config.scheduler.min_learning_rate = 0.1 * val
            print(f"Running with lr={val}, min_lr={config.scheduler.min_learning_rate}")
            my_train.main(config)
    elif args.command == "batch":
        for val in args.values:
            config.train.wandb.name = f"sweep_{args.command}_{val}"
            config.train.batch_size = int(val)
            print(f"Running with batch_size={val}")
            my_train.main(config)

if __name__ == "__main__":
    main()
