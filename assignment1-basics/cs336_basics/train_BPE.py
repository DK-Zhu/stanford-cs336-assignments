import os
import regex as re
import multiprocessing
from collections import Counter, defaultdict
import pickle

def find_chunk_boundaries(
    file_path: str | os.PathLike,
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )
    
    with open(file_path, "rb") as file:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(args):
    file_path, start, end, special_tokens = args
    with open(file_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

    pre_token_freq: defaultdict[tuple[bytes, ...], int] = defaultdict(int)
    
    delimiter = "|".join(map(re.escape, special_tokens))
    chunk = re.split(delimiter, chunk)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for text in chunk:
        matches = re.finditer(PAT, text)
        for match in matches:
            pre_token_bytes = tuple(bytes([i]) for i in match.group().encode("utf-8"))
            pre_token_freq[pre_token_bytes] += 1
    return pre_token_freq

def pre_tokenization(file_path: str | os.PathLike, special_tokens: list[str]):
    num_processes = multiprocessing.cpu_count()
    
    boundaries = find_chunk_boundaries(
        file_path=file_path, 
        desired_num_chunks=num_processes, 
        split_special_token="<|endoftext|>".encode("utf-8")
        )
        
    args = [(file_path, start, end, special_tokens) 
            for start, end in zip(boundaries[:-1], boundaries[1:])]
    with multiprocessing.Pool(num_processes) as pool:
        freqs = pool.map(process_chunk, args)

    pre_token_freq: Counter[tuple[bytes, ...]] = Counter()
    for freq in freqs:
        pre_token_freq.update(freq)
    return pre_token_freq

def get_most_frequent_BP(BP_freq: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    '''
    Given frequencies of byte pairs, return the most frequent byte pair.
    If ties occur, choose the lexicographically greater pair.
    '''
    chosen_BP: tuple[bytes, bytes] = (b'', b'')
    largest_freq = float('-inf')
    for BP, freq in BP_freq.items():
        if freq > largest_freq:
            largest_freq = freq
            chosen_BP = BP
        elif freq == largest_freq:
            if BP > chosen_BP:
                chosen_BP = BP
    return chosen_BP

def merge_pre_tokens(pre_token_freq, BP_to_merge):
    '''
    Update the pre_token_freq dict by merging the most frequent byte pair.
    '''
    new_pre_token_freq = {}
    for token_tuple, freq in pre_token_freq.items():
        new_token_tuple = []
        i = 0
        while i < len(token_tuple):
            if (
                i < len(token_tuple) - 1
                and token_tuple[i] == BP_to_merge[0]
                and token_tuple[i+1] == BP_to_merge[1]
            ):
                new_token_tuple.append(BP_to_merge[0] + BP_to_merge[1])
                i += 2
            else:
                new_token_tuple.append(token_tuple[i])
                i += 1
        new_pre_token_freq[tuple(new_token_tuple)] = freq
    pre_token_freq = new_pre_token_freq
    return pre_token_freq

def train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab = {i: bytes([i]) for i in range(256)}
    for id, token in enumerate(special_tokens, start=256):
        vocab[id] = token.encode("utf-8")

    pre_token_freq = pre_tokenization(input_path, special_tokens)

    # Build initial pair statistics and index
    BP_freq = defaultdict(int)
    BP_loc = defaultdict(set)  # pair -> set of token_tuples
    for token_tuple, freq in pre_token_freq.items():
        for i in range(len(token_tuple) - 1):
            BP = (token_tuple[i], token_tuple[i+1])
            BP_freq[BP] += freq
            BP_loc[BP].add(token_tuple)

    merges = []
    while len(vocab) < vocab_size:
        BP_to_merge = get_most_frequent_BP(BP_freq)
        merges.append(BP_to_merge)
        merged_BP = BP_to_merge[0] + BP_to_merge[1]
        vocab[len(vocab)] = merged_BP

        # Find all token_tuples containing BP_to_merge
        affected_tokens = list(BP_loc[BP_to_merge])
        BP_loc[BP_to_merge].clear()
        for token_tuple in affected_tokens:
            freq = pre_token_freq.pop(token_tuple)
            # Merge BP_to_merge in token_tuple
            new_token_tuple = []
            i = 0
            while i < len(token_tuple):
                if (
                    i < len(token_tuple) - 1
                    and token_tuple[i] == BP_to_merge[0]
                    and token_tuple[i+1] == BP_to_merge[1]
                ):
                    new_token_tuple.append(merged_BP)
                    i += 2
                else:
                    new_token_tuple.append(token_tuple[i])
                    i += 1
            new_token_tuple = tuple(new_token_tuple)
            pre_token_freq[new_token_tuple] = freq

            # Remove old pairs for this token_tuple
            for i in range(len(token_tuple) - 1):
                BP = (token_tuple[i], token_tuple[i+1])
                BP_freq[BP] -= freq
                BP_loc[BP].discard(token_tuple)

            # Add new pairs for new_token_tuple
            for i in range(len(new_token_tuple) - 1):
                BP = (new_token_tuple[i], new_token_tuple[i+1])
                BP_freq[BP] += freq
                BP_loc[BP].add(new_token_tuple)
    return (vocab, merges)

if __name__ == "__main__":
    file_path = "/Users/daokuan/Desktop/data/Datasets/owt_train.txt"
    vocab, merges = train_bpe_tokenizer(file_path, 32000, ['<|endoftext|>'])

    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    max_len = float("-inf")
    longest_token = None
    for tokenID, token in vocab.items():
        if len(token) > max_len:
            max_len = len(token)
            longest_token = token
    print("longest token:", longest_token)