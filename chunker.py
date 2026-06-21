from typing import List, Dict
import re


def _words(text: str) -> List[str]:
    # simple tokenizer by whitespace
    return re.findall(r"\S+", text)


def chunk_pages(pages: List[Dict], chunk_size_words: int = 500, overlap_words: int = 50) -> List[Dict]:
    """Chunk page texts into roughly `chunk_size_words` with `overlap_words` overlap.

    Each chunk is a dict: {"text": str, "page_numbers": [ints], "source": path}
    """
    chunks = []

    for page in pages:
        words = _words(page["text"]) or []
        total = len(words)
        start = 0
        while start < total:
            end = start + chunk_size_words
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page_numbers": [page["page_num"]],
                    "source": page.get("path"),
                })

            if end >= total:
                break
            start = end - overlap_words

    return chunks


def merge_nearby_chunks(chunks: List[Dict], max_pages_span: int = 3) -> List[Dict]:
    """Optional helper to merge consecutive chunks from the same document that span few pages.
    Keeps page_numbers unique and sorted.
    """
    if not chunks:
        return []
    merged = [chunks[0].copy()]
    for c in chunks[1:]:
        last = merged[-1]
        if c["source"] == last["source"] and (max(c["page_numbers"]) - min(last["page_numbers"]) <= max_pages_span):
            # merge
            last_text = last["text"] + "\n" + c["text"]
            last["text"] = last_text
            last["page_numbers"] = sorted(set(last["page_numbers"] + c["page_numbers"]))
        else:
            merged.append(c.copy())
    return merged
