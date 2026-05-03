"""

Usage:
  tokenizer_experiments.py (calc_compression_ratio | encode_file) [--vocab=<FILE>] [--merge=<FILE>] [--input=<INPUT>] [--output=<OUTPUT>]

Arguments:

Options:
  -h --help                     Show this screen.
  --vocab=<FILE>                vocab file [default: output/tinystories_vocab.json]
  --merge=<FILE>                merge file [default: output/tinystories_merges.txt]
  --input=<INPUT>               input file [default: data/TinyStoriesV2-GPT4-valid.txt]
  --output=<OUTPUT>             output file [default: output/tinystories_valid.bin]

Example:
  tokenizer_experiments.py calc_compression_ratio --vocab=output/owt_vocab.json --merge=output/owt_merges.txt --input=data/owt_valid.txt --output=output/owt_valid.bin

"""

import docopt
import logging
import numpy as np
import os
from cs336_basics.tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO)

special_token = "<|endoftext|>"

def sample_file(file, n=10):
    fsize = os.path.getsize(file)
    sample: list[str] = []
    idxes: list[int] = sorted(np.random.randint(0, fsize, size=n*2))
    with open(file, 'rb') as f:
        for i in np.arange(0, n*2, 2):
            f.seek(idxes[i])
            chunk = f.read(idxes[i+1]-idxes[i])
            sample.append(chunk.decode('utf-8', errors='ignore'))
    return sample

def calc_compression_ratio(tokenizer: Tokenizer, samples: list[str]):
    total_bytes = 0
    total_tokens = 0
    for sample in samples:
        ids = tokenizer.encode(sample)
        total_bytes += len(sample.encode('utf-8'))
        total_tokens += len(ids)
    return total_bytes / total_tokens, total_bytes, total_tokens

def main():
    arguments = docopt.docopt(__doc__)
    logging.info(arguments)
    tokenizer = Tokenizer.from_files(arguments['--vocab'], arguments['--merge'])

    if arguments['calc_compression_ratio']:
        logging.info(f'calculating compression ratio for {arguments["--input"]} with {arguments["--vocab"]} and {arguments["--merge"]}')
        samples = sample_file(arguments['--input'], n=10)
        ratio = calc_compression_ratio(tokenizer, samples)
        print(f"compression ratio for {arguments["--input"]} with {arguments["--vocab"]} and {arguments["--merge"]}: {ratio[0]:.4f}")
    elif arguments['encode_file']:
        logging.info(f'tokenizing input {arguments["--input"]}, saving to {arguments["--output"]}')
        with open(arguments['--input'],"r", encoding="utf-8") as input, \
            open(arguments['--output'],"wb") as output:
            buffer = np.empty(10_000_000, dtype=np.uint16)
            n = 0
            for id in tokenizer.encode_iterable(input):
                if id > np.iinfo(np.uint16).max or id < np.iinfo(np.uint16).min:
                    raise ValueError(f"id {id} is out of range for uint16")
                buffer[n] = id
                n += 1
                if n == 10_000_000:
                    output.write(buffer.tobytes())
                    n = 0
            output.write(buffer[:n].tobytes())

if __name__ == '__main__':
    main()