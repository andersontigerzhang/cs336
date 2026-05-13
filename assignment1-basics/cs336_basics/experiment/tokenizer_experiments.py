import argparse
import torch
import logging
import numpy as np
import os
from cs336_basics.tokenizer import Tokenizer
from cs336_basics import config, my_data, my_nn_models, my_nn_functions

logging.basicConfig(level=logging.INFO)

special_token = "<|endoftext|>"

def sample_file(file, n=10):
    fsize = os.path.getsize(file)
    sample: list[str] = []
    idxes: list[int] = sorted(np.random.randint(0, fsize, size=n * 2))
    with open(file, "rb") as f:
        for i in np.arange(0, n * 2, 2):
            f.seek(idxes[i])
            chunk = f.read(idxes[i + 1] - idxes[i])
            sample.append(chunk.decode("utf-8", errors="ignore"))
    return sample


def calc_compression_ratio(tokenizer: Tokenizer, samples: list[str]):
    total_bytes = 0
    total_tokens = 0
    for sample in samples:
        ids = tokenizer.encode(sample)
        total_bytes += len(sample.encode("utf-8"))
        total_tokens += len(ids)
    return total_bytes / total_tokens, total_bytes, total_tokens


def calc_compression(tokenizer: Tokenizer, args):
    logging.info(f"calculating compression ratio for {args.input} with {args.vocab} and {args.merge}")
    samples = sample_file(args.input, n=10)
    return calc_compression_ratio(tokenizer, samples)
    # print(f"compression ratio for {args.input} with {args.vocab} and {args.merge}: {ratio[0]:.4f}")


def encode_file(tokenizer: Tokenizer, args):
    logging.info(f"tokenizing input {args.input}, saving to {args.output}")
    with open(args.input, "r", encoding="utf-8") as input_file, open(args.output, "wb") as output:
        buffer = np.empty(10_000_000, dtype=np.uint16)
        n = 0
        for id in tokenizer.encode_iterable(input_file):
            if id > np.iinfo(np.uint16).max or id < np.iinfo(np.uint16).min:
                raise ValueError(f"id {id} is out of range for uint16")
            buffer[n] = id
            n += 1
            if n == 10_000_000:
                output.write(buffer.tobytes())
                n = 0
        output.write(buffer[:n].tobytes())

def decode_file(tokenizer: Tokenizer, args):
    logging.info(f"decoding input {args.input}")

    cfg = config.get_config(os.path.join(args.model_path, "config.yaml"))
    model = my_nn_models.TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta,
        device=torch.device(cfg.model.device if "cuda" in cfg.model.device and torch.cuda.is_available() else "cpu"),
        dtype=getattr(torch, cfg.model.dtype),
    )
    ckpt = os.path.join(args.model_path, "checkpoint.pt")
    if os.path.exists(ckpt):
        my_data.load_checkpoint(os.path.join(args.model_path, "checkpoint.pt"), model)
    ids = tokenizer.encode(args.input)
    tokens = torch.empty((1,args.max_len+1), dtype=torch.int32)
    tokens[0,np.arange(len(ids))]=torch.tensor(ids,dtype=torch.int32)
    topk = 1 if args.topk is None else args.topk
    special = tokenizer.encode(special_token)
    token_len = len(ids)
    with torch.no_grad():
        for i in range(token_len, args.max_len):
            logits = model(tokens[:,:token_len]) / args.temperature
            topk_logits, topk_indices = torch.topk(logits[:,-1,:], topk, dim=-1)    
            probs = my_nn_functions.softmax(topk_logits, dim=-1)
            token = topk_indices.gather(-1, torch.multinomial(probs, 1))
            tokens[:, i] = token.squeeze(-1)
    return tokenizer.decode(tokens[0,:i+1].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    parser_calc = subparsers.add_parser("calc_compression", help="Calculate compression ratio")
    parser_calc.add_argument("--vocab", type=str, default="output/tinystories_vocab.json", help="Vocab file")
    parser_calc.add_argument("--merge", type=str, default="output/tinystories_merges.txt", help="Merge file")
    parser_calc.add_argument("--input", type=str, default="data/TinyStoriesV2-GPT4-valid.txt", help="Input file")

    parser_encode = subparsers.add_parser("encode_file", help="Encode text file to tokens")
    parser_encode.add_argument("--vocab", type=str, default="output/tinystories_vocab.pkl", help="Vocab file")
    parser_encode.add_argument("--merge", type=str, default="output/tinystories_merges.pkl", help="Merge file")
    parser_encode.add_argument("--input", type=str, default="data/TinyStoriesV2-GPT4-valid.txt", help="Input text file")
    parser_encode.add_argument(
        "--output", type=str, default="output/tinystories_valid.bin", help="Output binary token file"
    )

    parser_decode = subparsers.add_parser("decode", help="Decode with model")
    parser_decode.add_argument("--vocab", type=str, default="output/tinystories_vocab.json", help="Vocab file")
    parser_decode.add_argument("--merge", type=str, default="output/tinystories_merges.txt", help="Merge file")
    parser_decode.add_argument("--input", type=str, default="Once upon a time", help="Input file")
    parser_decode.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser_decode.add_argument("--topk", type=int, default=3, help="Top-k sampling (default: 3)")
    parser_decode.add_argument("--max_len", type=int, default=100, help="Top-k sampling (default: 3)")
    parser_decode.add_argument(
        "--temperature", type=float, default=0.75, help="Temperature for sampling (default: 0.75)"
    )

    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merge)

    if args.command == "calc_compression":
        ratio, total_bytes, total_tokens = calc_compression(tokenizer, args)
        logging.info(f"compression ratio for {args.input} with {args.vocab} and {args.merge}: {ratio:.4f}")
    elif args.command == "encode_file":
        encode_file(tokenizer, args)
    elif args.command == "decode":
        decoded = decode_file(tokenizer, args)
        logging.info(f"Prompt: {args.input}")
        logging.info(f"Decoded:\n{decoded}")
    else:
        parser.print_help()
