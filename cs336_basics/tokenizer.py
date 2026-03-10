from __future__ import annotations

import heapq
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class DLLNode:
    __slots__ = ["v", "prev", "next"]

    def __init__(self, v):
        self.v = v
        self.prev: "DLLNode | None" = None
        self.next: "DLLNode | None" = None


class HeapItem:
    __slots__ = ["count", "token_id1", "token_id2", "vocab_ref", "bytes1", "bytes2"]

    def __init__(self, count, token_id1, token_id2, vocab_ref):
        self.count = count
        self.token_id1 = token_id1
        self.token_id2 = token_id2
        self.vocab_ref = vocab_ref
        self.bytes1 = vocab_ref[token_id1]
        self.bytes2 = vocab_ref[token_id2]

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count
        if self.bytes1 != other.bytes1:
            return self.bytes1 > other.bytes1
        return self.bytes2 > other.bytes2

    def get_pair(self):
        return self.token_id1, self.token_id2


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs, strict=False))


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8", errors="replace")

    if special_tokens:
        escaped_tokens = [re.escape(t) for t in special_tokens]
        parts = re.split("(" + "|".join(escaped_tokens) + ")", text)
    else:
        parts = [text]

    gpt2_pat = re.compile(PAT)
    words_list: list[list[int]] = []
    special_set = set(special_tokens)
    for part in parts:
        if part in special_set:
            continue
        for word in gpt2_pat.findall(part):
            words_list.append([b for b in word.encode("utf-8")])

    head_nodes = []
    stats = defaultdict(int)
    indices: dict[tuple[int, int], list[DLLNode]] = defaultdict(list)

    for word in words_list:
        if not word:
            continue

        head = DLLNode(word[0])
        head_nodes.append(head)
        prev = head
        for i in range(1, len(word)):
            curr = DLLNode(word[i])
            prev.next = curr
            curr.prev = prev

            pair = (prev.v, curr.v)
            stats[pair] += 1
            indices[pair].append(prev)
            prev = curr

    vocab = {i: bytes([i]) for i in range(256)}
    pq: list[HeapItem] = []
    for pair, count in stats.items():
        heapq.heappush(pq, HeapItem(count, pair[0], pair[1], vocab))

    merges: list[tuple[bytes, bytes]] = []
    num_merges = max(0, vocab_size - 256 - len(special_tokens))

    for i in range(num_merges):
        best_pair = None
        while pq:
            item = heapq.heappop(pq)
            cur_pair = item.get_pair()
            if item.count != stats.get(cur_pair, 0):
                continue
            best_pair = cur_pair
            break

        if best_pair is None:
            break

        new_id = 256 + i
        left, right = best_pair
        merges.append((vocab[left], vocab[right]))
        vocab[new_id] = vocab[left] + vocab[right]

        nodes = indices[best_pair]
        del stats[best_pair]
        del indices[best_pair]

        for node in nodes:
            if node.v != left or node.next is None or node.next.v != right:
                continue

            prev_node = node.prev
            next_node = node.next
            next_next_node = next_node.next

            if prev_node:
                old_pair = (prev_node.v, node.v)
                if old_pair != best_pair:
                    stats[old_pair] -= 1
                    if stats[old_pair] == 0:
                        del stats[old_pair]
                    else:
                        heapq.heappush(pq, HeapItem(stats[old_pair], prev_node.v, node.v, vocab))

                new_pair = (prev_node.v, new_id)
                stats[new_pair] += 1
                indices[new_pair].append(prev_node)
                heapq.heappush(pq, HeapItem(stats[new_pair], prev_node.v, new_id, vocab))

            if next_next_node:
                old_pair = (next_node.v, next_next_node.v)
                if old_pair != best_pair:
                    stats[old_pair] -= 1
                    if stats[old_pair] == 0:
                        del stats[old_pair]
                    else:
                        heapq.heappush(pq, HeapItem(stats[old_pair], next_node.v, next_next_node.v, vocab))

                new_pair = (new_id, next_next_node.v)
                stats[new_pair] += 1
                indices[new_pair].append(node)
                heapq.heappush(pq, HeapItem(stats[new_pair], new_id, next_next_node.v, vocab))

            node.v = new_id
            node.next = next_next_node
            if next_next_node:
                next_next_node.prev = node
            next_node.v = -1

    next_id = 256 + len(merges)
    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1

    return vocab, merges


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.encoder = {v: k for k, v in self.vocab.items()}
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.pat = re.compile(PAT)

        if special_tokens:
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(t) for t in sorted_special_tokens]
            self.special_pat = re.compile(r"(" + "|".join(escaped_tokens) + r")")
        else:
            self.special_pat = None

        cur_max_id = max(self.vocab.keys()) if self.vocab else 255
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.encoder:
                cur_max_id += 1
                self.vocab[cur_max_id] = st_bytes
                self.encoder[st_bytes] = cur_max_id

    def decode(self, ids: list[int]) -> str:
        byte_parts = [self.vocab[i] for i in ids if i in self.vocab]
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def _bpe(self, token_bytes: bytes) -> list[int]:
        tokens = [bytes([b]) for b in token_bytes]
        prev = None
        head = None
        for token in tokens:
            curr = DLLNode(token)
            curr.prev = prev
            if prev is not None:
                prev.next = curr
            else:
                head = curr
            prev = curr

        while True:
            curr = head
            best_pair = None
            lowest_rank = float("inf")
            while curr is not None and curr.next is not None:
                curr_pair = (curr.v, curr.next.v)
                curr_rank = self.bpe_ranks.get(curr_pair, float("inf"))
                if curr_rank < lowest_rank:
                    best_pair = curr_pair
                    lowest_rank = curr_rank
                curr = curr.next

            if lowest_rank == float("inf"):
                break

            curr = head
            while curr is not None and curr.next is not None:
                curr_pair = (curr.v, curr.next.v)
                next_node = curr.next
                if curr_pair == best_pair:
                    curr.v = curr.v + next_node.v
                    curr.next = next_node.next
                    if next_node.next is not None:
                        next_node.next.prev = curr
                curr = curr.next

        ids: list[int] = []
        curr = head
        while curr is not None:
            ids.append(self.encoder[curr.v])
            curr = curr.next
        return ids

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            if self.special_pat:
                parts = self.special_pat.split(text_chunk)
            else:
                parts = [text_chunk]

            for part in parts:
                part_bytes = part.encode("utf-8")
                if part in self.special_tokens and part_bytes in self.encoder:
                    yield self.encoder[part_bytes]
                else:
                    for token in self.pat.findall(part):
                        yield from self._bpe(token.encode("utf-8"))

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        byte_encoder = bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_reversed = json.load(f)

        vocab = {}
        for token_str, token_id in vocab_reversed.items():
            token_bytes = bytes([byte_decoder[c] for c in token_str])
            vocab[token_id] = token_bytes

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            lines = f.read().splitlines()

        start_idx = 1 if lines and lines[0].startswith("#version") else 0
        for line in lines[start_idx:]:
            if not line.strip():
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                continue
            token_bytes1 = bytes([byte_decoder[c] for c in parts[0]])
            token_bytes2 = bytes([byte_decoder[c] for c in parts[1]])
            merges.append((token_bytes1, token_bytes2))

        return cls(vocab, merges, special_tokens)
