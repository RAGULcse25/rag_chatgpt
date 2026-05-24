# ingest.py
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, RateLimitError
from pypdf import PdfReader
import numpy as np
import faiss

load_dotenv()
client = OpenAI()

DOCS_DIR = Path("docs")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"

def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)

def load_documents():
    texts = []
    for path in DOCS_DIR.glob("*"):
        if path.suffix.lower() in [".txt", ".md"]:
            texts.append((path.name, read_text_file(path)))
        elif path.suffix.lower() == ".pdf":
            texts.append((path.name, read_pdf_file(path)))
    return texts

def chunk_text(text, chunk_size=800, overlap=150):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def local_embed_texts(texts):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Local embeddings require the sentence-transformers package. Install with:\n"
            "    pip install sentence-transformers"
        ) from e
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.array(embs, dtype="float32")


def embed_texts(texts, batch_size=32, use_local_on_fail=True):
    all_embeddings = []
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            res = client.embeddings.create(model=EMBED_MODEL, input=batch)
            all_embeddings.extend(item.embedding for item in res.data)
        return np.array(all_embeddings, dtype="float32")
    except RateLimitError as e:
        if use_local_on_fail:
            print("OpenAI rate/quotas exhausted — falling back to local embeddings (if available)...")
            return local_embed_texts(texts)
        raise RuntimeError(
            "OpenAI quota or rate limit error: check your OpenAI plan, billing, and usage. "
            "If your account quota is exhausted, you must upgrade or reset usage before retrying."
        ) from e
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI embeddings request failed: {e}") from e

def main():
    docs = load_documents()
    all_chunks = []
    meta = []

    for filename, text in docs:
        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append(chunk)
            meta.append({"source": filename, "chunk_id": i})

    if not all_chunks:
        raise ValueError("No documents found in docs/")

    try:
        vectors = embed_texts(all_chunks)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return

    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    with open(DATA_DIR / "index.pkl", "wb") as f:
        pickle.dump(
            {
                "index": index,
                "chunks": all_chunks,
                "meta": meta,
                "dim": dim,
                "model": EMBED_MODEL,
            },
            f,
        )

    print(f"Saved {len(all_chunks)} chunks to data/index.pkl")

if __name__ == "__main__":
    main()