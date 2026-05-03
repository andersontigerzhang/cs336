import os
from io import BytesIO
from typing import BinaryIO
from cs336_basics.utils import text_to_pretoks, init_regex
from concurrent import futures
from functools import reduce
import logging
from collections import Counter,defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries

def process_chunk(file: BinaryIO | str, start: int, end: int, special_tokens: list[str]) -> Counter:
    """
    Process a chunk of the file from start to end, returning a Counter of token frequencies.
    """
    if isinstance(file, str):
        file = open(file, "rb")
    file.seek(start)
    text = file.read(end - start).decode("utf-8", errors="ignore")
    tokens = text_to_pretoks(special_tokens, text)
    return Counter(tokens)

def build_byte_pair_frequencies(token_counts: Counter) -> Counter:
    """
    Given a Counter of token frequencies, build a Counter of byte pair frequencies.
    """
    pair_freqs = Counter()
    pair_to_tok: defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for token, freq in token_counts.items():
        if len(token) < 2:
            continue
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_freqs[pair] += freq
            pair_to_tok[pair].add(token)
    return pair_freqs, pair_to_tok

def merge_pair(word, pair, replacement):
    result = []
    i = 0
    word_len = len(word)
    while i < word_len:
        if i < word_len - 1 and (word[i], word[i+1]) == pair:
            result.append(replacement)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)

def change_byte_pair_frequencies(pair_freqs: Counter, 
                                 pair_to_tok: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]], 
                                 pair: tuple[bytes, bytes], add=True):
    for token in pair_to_tok[pair]:
        pair_cnts = Counter(zip(token, token[1:]))
        if add:
            pass
        else:
            pass

def train_bpe(input_path: str | os.PathLike, 
              vocab_size: int,
              special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input file, returning a dictionary mapping tokens to their ranks.
    """
    # Placeholder implementation: In a real implementation, you would read the input file,
    # count token frequencies, and iteratively merge the most frequent pairs until you reach vocab_size.
    # For simplicity, we'll just return a dummy vocabulary here.
    vocab: dict[int, bytes] = {i : bytes([i]) for i in range(256)}
    current_vocab_size = len(vocab)
    for token in special_tokens:
        vocab[current_vocab_size] = token.encode("utf-8")
        current_vocab_size += 1

    merges: list[tuple[bytes, bytes]] = []

    num_workers = min(os.cpu_count() or 1, 4)  # Use up to 4 workers

    fsize = os.path.getsize(input_path)
    num_chunks = max(1, fsize // (1 * 1024 * 1024))  # Aim for ~10MB per chunk
    chunks = find_chunk_boundaries(open(input_path, "rb"), desired_num_chunks=num_chunks, split_special_token=special_tokens[0].encode("utf-8"))
    if num_chunks == 1:
        logging.info(f"Processing file in a single chunk...")
        init_regex()
        pretoks = process_chunk(open(input_path, "rb"), 0, fsize, special_tokens)
    else:
        # jobs = [(str(input_path), start, end, special_tokens) for start, end in zip(chunks[:-1], chunks[1:])]
        # with Pool(processes=num_workers, initializer=init_regex) as pool:
        #     results = pool.starmap(process_chunk, jobs)
        logging.info(f"Processing {len(chunks)-1} chunks with {num_workers} workers...")
        with futures.ProcessPoolExecutor(max_workers=num_workers, initializer=init_regex) as executor:
            futures_list = [executor.submit(process_chunk, str(input_path), start, end, special_tokens) for start, end in zip(chunks[:-1], chunks[1:])]
            results = [f.result() for f in futures_list]
        pretoks = reduce(lambda d, src: d.update(src) or d, results, Counter())
    
    pair_freqs, pair_to_tok = build_byte_pair_frequencies(pretoks)
    while current_vocab_size < vocab_size:
        if not pair_freqs:
            break
        cnt = max(pair_freqs.values())
        pair = max([p for p, c in pair_freqs.items() if c == cnt])

        merges.append(pair)
        vocab[current_vocab_size] = b"".join(pair)
        current_vocab_size += 1
        tokens_to_merge = pair_to_tok[pair].copy() # tokens that contain the pair to merge
        add_back: defaultdict[tuple[bytes, ...], int] = defaultdict(int)
        for token in tokens_to_merge:
            pairs_in_token = Counter(zip(token, token[1:]))
            for k,v in pairs_in_token.items():
                pair_to_tok[k].discard(token)
                if not pair_to_tok[k]:
                    pair_to_tok.pop(k, None)
                
                # # every pairs in the token to merge will have its frequency reduced 
                # # by the count of the token, since those pairs will be replaced by the merged token
                updated = pair_freqs.get(k, 0) - v * pretoks[token]
                if updated > 0:
                    pair_freqs[k] = updated
                else:
                    pair_freqs.pop(k, None)
            merged_token = merge_pair(token, pair, b"".join(pair))
            add_back[merged_token] += pretoks[token]
            pretoks[merged_token] = pretoks.pop(token)
        for k,v in add_back.items():
            pairs_in_k = Counter(zip(k, k[1:]))
            for pk in pairs_in_k:
                pair_to_tok[pk].add(k)
                pair_freqs[pk] = pair_freqs.get(pk, 0) + v * pairs_in_k[pk]
        # pair_freqs1, pair_to_tok1 = build_byte_pair_frequencies(pretoks)
    return vocab, merges

if __name__ == "__main__":
    # vocab, merges = train_bpe("tests/fixtures/corpus.en", vocab_size=1000, special_tokens=["<|endoftext|>"])
    vocab = train_bpe("/tmp/a", vocab_size=10000, special_tokens=["<|endoftext|>", '\n', ' ', '\t'])