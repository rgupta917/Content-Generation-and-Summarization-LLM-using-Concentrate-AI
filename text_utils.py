from typing import List


def chunk_text(text: str, max_chars: int = 8000, overlap: int = 300) -> List[str]:
    """
    Simple char-based chunker.
    max_chars keeps prompts under typical context limits.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def safe_filename(s: str) -> str:
    s = s.strip().lower()
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("-")
    out = "".join(keep).strip("-")
    return out[:80] if out else "output"
