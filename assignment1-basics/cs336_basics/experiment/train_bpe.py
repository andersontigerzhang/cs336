"""

Usage:
  train_bpe_expts_owt.py [--vocab_size=<vocab_size>] [--data=<data>] [--output_vocab=<output_vocab>] [--output_merges=<output_merges>]

Arguments:

Options:
  -h --help                     Show this screen.
  --vocab_size=<n>              vocab size [default: 10000]
  --data=<data>            input file [default: data/TinyStoriesV2-GPT4-train.txt]
  --output_vocab=<out>     output for vocab [default: output/tinystories_vocab.pkl]
  --output_merges=<out>    output for merges [default: output/tinystories_merges.pkl]

Example:
  train_bpe_expts_owt.py --vocab_size=10000 --data=data/owt_train.txt --output_vocab=output/owt_vocab.json --output_merges=output/owt_merges.txt
"""

from cs336_basics.train_bpe import train_bpe
# from cs336_basics.utils import save_voacb_and_merge
import docopt
import cProfile
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)

    vocab_size = int(arguments['--vocab_size'])
    data = arguments['--data']
    output_vocab = arguments['--output_vocab']
    output_merges = arguments['--output_merges']

    logging.info(arguments)
    prof = cProfile.Profile()
    prof.enable()
    vocab, merges = train_bpe(data, vocab_size, special_tokens=["<|endoftext|>"])
    prof.disable()
    prof.dump_stats('train_bpe_owt.prof')
    # prof.sort_stats('tottime').print_stats(20)
    # save_voacb_and_merge(vocab, merges, output_vocab, output_merges)
    with open(output_vocab, 'wb') as f:
      pickle.dump(vocab, f)
    with open(output_merge, 'wb') as f:
      pickle.dump(merge, f)

