from typing import Iterable
import os
# from cs336_basics.utils import get_tokenizer_from_vocab_merges_path
from cs336_basics.train_bpe import text_to_pretoks, init_regex, merge_pair
from io import BytesIO
from functools import lru_cache

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.vocab = vocab
        self.bytes2id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self._merge_ranks = {merge: rank for rank, merge in enumerate(merges)}
        self.special_tokens = special_tokens or ['<|endoftext|>']

    @classmethod
    def from_files(cls: type, vocab_path: str | os.PathLike, merges_path: str | os.PathLike, special_tokens: list[str] = None):
        # vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_path, merges_path)
        # return cls(vocab, merges, special_tokens=special_tokens)
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        with open(merge_path_path, 'rb') as f:
            merge = pickle.load(f)
        return cls(vocab=vocab,merge=merge,special_tokens=special_tokens)
    
    def encode(self, text: str) -> list[int]:
        # import tracemalloc
        # tracemalloc.start()
        init_regex()
        pretoks = text_to_pretoks(self.special_tokens, text, include_special_tokens=True)
        ids: list[int] = []
        for pretok in pretoks:
            pretok = self.encode_pretok(pretok)
            ids.extend([self.bytes2id[tok] for tok in pretok])
        # current, peak = tracemalloc.get_traced_memory()
        # tracemalloc.stop()
        # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        return ids
    
    @lru_cache(maxsize=10_000_000)
    def encode_pretok(self, pretok):
        while len(pretok) > 1:
            pairs = zip(pretok, pretok[1:])
            best_rank = float('inf')
            best_pair = None
            for pair in pairs:
                if pair in self._merge_ranks:
                    rank = self._merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
            if best_pair is None:
                break
            else:
                pretok = merge_pair(pretok, best_pair, b''.join(best_pair))
        return pretok

    def encode_iterable(self, texts: Iterable[str]) -> Iterable[list[int]]:
        for text in texts:
            for id in self.encode(text):
                yield id

    def decode(self, tokens: list[int]) -> str:
        results: bytes = b''
        for token in tokens:
            results += self.vocab[token]
        return results.decode("utf-8", errors="ignore")

if __name__ == "__main__":
    # tokenizer = Tokenizer(vocab={0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'},
    #                       merges=[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a',b't')])
    # print(tokenizer.encode("the cat ate"))
    FIXTURES_PATH = 'tests/fixtures'
    tiny = FIXTURES_PATH / 'tinystories_sample_5M.txt'
    VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
    MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

    tokenizer = Tokenizer.from_files(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"])
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        contents = f.read()
        ids = tokenizer.encode(contents)
        # printtokenizer, contents)