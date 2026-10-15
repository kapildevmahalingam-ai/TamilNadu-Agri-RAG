"""Extracts PDFs, splits text, generates local open-source embeddings,
and saves a searchable FAISS index with PDF/page citations.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
import faiss
import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
INDEX_DIR = BASE_DIR / "faiss_index"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 2  # Micro-batching for large 15MB PDFs

def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    pages = []
    try:
        with pymupdf.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text", sort=True).strip()
                if text:
                    pages.append({"text": text, "source": pdf_path.name, "page": page_number})
    except Exception as exc:
        raise RuntimeError(f"Could not read PDF '{pdf_path.name}': {exc}") from exc
    return pages

def split_text(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + int(CHUNK_SIZE * 0.65), end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks

def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory does not exist: {DATA_DIR}")
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDF files found directly inside: {DATA_DIR}")

    records = []
    for pdf_path in pdf_files:
        print(f"\nExtracting: {pdf_path.name}")
        pages = extract_pdf_pages(pdf_path)
        for page_data in pages:
            chunks = split_text(page_data["text"])
            for chunk_number, chunk in enumerate(chunks, start=1):
                records.append({
                    "text": chunk,
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunk": chunk_number
                })
        print(f"  Pages with text: {len(pages)}")

    if not records:
        raise SystemExit("No text chunks were created.")

    print(f"\nTotal text chunks created: {len(records)}")
    print(f"Loading local open embedding model ({EMBED_MODEL_NAME})...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Generating embeddings via micro-batches...")
    all_embeddings = []
    total = len(records)
    
    for start in range(0, total, BATCH_SIZE):
        batch_records = records[start:start + BATCH_SIZE]
        batch_texts = [item["text"] for item in batch_records]
        batch_number = start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"Processing micro-batch {batch_number}/{total_batches} ({len(batch_texts)} chunks)...")
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.extend(batch_embeddings)
        time.sleep(0.05)

    matrix = np.asarray(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / "index.faiss"
    chunks_path = INDEX_DIR / "chunks.jsonl"

    faiss.write_index(index, str(index_path))
    with chunks_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 55)
    print("MICRO-BATCH PDF INGESTION COMPLETED SUCCESSFULLY[cite: 2]")
    print("=" * 55)
    print(f"Chunks indexed: {index.ntotal}")
    print(f"FAISS index: {index_path}")
    print(f"Chunk metadata: {chunks_path}")

if __name__ == "__main__":
    main()