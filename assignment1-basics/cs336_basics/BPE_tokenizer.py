import regex as re
import pickle
from typing import Iterator, Iterable
import time
import numpy as np
from cs336_basics.text_splitter import find_chunk_boundaries
import multiprocessing
import os

class Tokenizer:
    def __init__(
            self, 
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]], 
            special_tokens: list[str] | None = None
    ) -> None:
        self.id_to_token = vocab
        self.token_to_id = {token: id for id, token in vocab.items()}
        self.merges = merges

        if special_tokens:
            self.special_tokens = set(special_tokens)  # for quick look-up
            for special_token in self.special_tokens:
                b = special_token.encode("utf-8")
                if b not in self.token_to_id:
                    new_id = max(self.token_to_id.values()) + 1
                    self.token_to_id[b] = new_id
                    self.id_to_token[new_id] = b
        else:
            self.special_tokens = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, "rb") as vf:
            vocab = pickle.load(vf)
        with open(merges_filepath, "rb") as mf:
            merges = pickle.load(mf)
        return cls(vocab, merges, special_tokens)
    
    def _encode_part(self, part: str) -> Iterator[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        matches = re.finditer(PAT, part)
        for match in matches:
            pre_token = list(bytes([i]) for i in match.group().encode("utf-8"))
            BP_set = set((pre_token[i], pre_token[i+1]) for i in range(len(pre_token)-1)) # for quick look-up
            for merge in self.merges:
                if merge in BP_set:
                    new_pre_token = []
                    i = 0
                    while i < len(pre_token):
                        if (
                            i < len(pre_token) - 1
                            and pre_token[i] == merge[0]
                            and pre_token[i+1] == merge[1]
                        ):
                            new_pre_token.append(pre_token[i] + pre_token[i+1])
                            i += 2
                        else:
                            new_pre_token.append(pre_token[i])
                            i += 1
                    pre_token = new_pre_token
                    BP_set = set((pre_token[i], pre_token[i+1]) for i in range(len(pre_token)-1))
            for b in pre_token:
                yield self.token_to_id[b]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = [] # encoded IDs
        
        if self.special_tokens:
            delimiter = "(" + "|".join(map(
                re.escape, 
                sorted(self.special_tokens, key=lambda x: -len(x))
            )) + ")"
            parts = re.split(delimiter, text)
        
            for part in parts:
                if part in self.special_tokens:
                    ids.append(self.token_to_id[part.encode("utf-8")])
                else:
                    for encoded_id in self._encode_part(part):
                        ids.append(encoded_id)
        else:
            for encoded_id in self._encode_part(text):
                ids.append(encoded_id)
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        '''
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token[id] for id in ids]
        return b''.join(tokens).decode("utf-8", errors="replace")
    
def encode_chunk(args):
    tokenizer, file_path, start, end = args
    with open(file_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end-start).decode("utf-8", errors="ignore")
    return tokenizer.encode(chunk)

if __name__ == "__main__":
    parent_folder = "/Users/daokuan/Desktop/data"
    special_tokens = ['<|endoftext|>']
    num_processes = multiprocessing.cpu_count()

    # tokenizer for tiny stories
    vocab_filepath = f"{parent_folder}/Tokenizers/TinyStoriesV2-GPT4-train/vocab.pkl"
    merges_filepath = f"{parent_folder}/Tokenizers/TinyStoriesV2-GPT4-train/merges.pkl"
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    # # tokenizer for owt
    # vocab_filepath = f"{parent_folder}/Tokenizers/owt_train/vocab.pkl"
    # merges_filepath = f"{parent_folder}/Tokenizers/owt_train/merges.pkl"
    # tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    filepath = f"{parent_folder}/Datasets/TinyStoriesV2-GPT4-train.txt"
    # filepath = f"{parent_folder}/Datasets/TinyStoriesV2-GPT4-valid.txt"
    # owt_filepath = f"{parent_folder}/Datasets/owt_train.txt"
    # owt_filepath = f"{parent_folder}/Datasets/owt_valid.txt"

    boundaries = find_chunk_boundaries(
        file_path=filepath,
        desired_num_chunks=num_processes,
        split_special_token="<|endoftext|>".encode("utf-8")
    )
    args = [(tokenizer, filepath, start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:])]
    with multiprocessing.Pool(num_processes) as pool:
        token_ids = np.array([token_id for chunk in pool.map(encode_chunk, args) for token_id in chunk], dtype=np.uint16)

    np.save(f"{parent_folder}/Datasets/TinyStories-tokenIDs-train.npy", token_ids)
    # np.save(f"{parent_folder}/Datasets/TinyStories-tokenIDs-valid.npy", token_ids)
    # np.save(f"{parent_folder}/Datasets/owt-tokenIDs-train.npy", token_ids)
    # np.save(f"{parent_folder}/Datasets/owt-tokenIDs-valid.npy", token_ids)
    
    # # compute throughput and compression ratio
    # start_time = time.time()
    # with multiprocessing.Pool(num_processes) as pool:
    #     token_ids = np.array([token_id for chunk in pool.map(encode_chunk, args) for token_id in chunk], dtype=np.uint16)
    # elapsed = time.time() - start_time
    # with open(filepath, "rb") as file:
    #     file.seek(0, os.SEEK_END)
    #     file_size = file.tell()
    # print(f"throughput: {file_size / elapsed} bytes/second")
    # print("compression ratio (tiny stories - valid):", file_size / len(token_ids))
