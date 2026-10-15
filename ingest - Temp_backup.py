
"""
Tamil Nadu Agriculture Schemes RAG - PDF Ingestion
Extract PDFs, split text into chunks, generate Gemini embeddings,
and save a searchable FAISS index with PDF/page citations.
"""

import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
import pymupdf
from dotenv import load_dotenv
from google import genai
from google.genai import types


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
INDEX_DIR = BASE_DIR / "faiss_index"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
)

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 220

# Number of chunks sent in one embedding request.
BATCH_SIZE = 64

# Retry transient API errors.
MAX_RETRIES = 5


# --------------------------------------------------
# 2. Extract text from PDF files
# --------------------------------------------------

def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract selectable text and preserve PDF page numbers."""

    pages = []

    try:
        with pymupdf.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text", sort=True).strip()

                if text:
                    pages.append({
                        "text": text,
                        "source": pdf_path.name,
                        "page": page_number,
                    })

    except Exception as exc:
        raise RuntimeError(
            f"Could not read PDF '{pdf_path.name}': {exc}"
        ) from exc

    return pages


# --------------------------------------------------
# 3. Split page text into overlapping chunks
# --------------------------------------------------

def split_text(text: str) -> list[str]:
    """Split text while preserving overlap between chunks."""

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Prefer a word boundary where possible.
        if end < len(text):
            boundary = text.rfind(
                " ",
                start + int(CHUNK_SIZE * 0.65),
                end,
            )

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


# --------------------------------------------------
# 4. Generate embeddings using Gemini
# --------------------------------------------------

def embed_batch(
    client: genai.Client,
    texts: list[str],
) -> list[list[float]]:
    """Generate document embeddings using the Gemini API."""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                ),
            )

            if not response.embeddings:
                raise RuntimeError(
                    "Gemini returned no embeddings."
                )

            vectors = [
                embedding.values
                for embedding in response.embeddings
            ]

            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Expected {len(texts)} embeddings, "
                    f"but received {len(vectors)}."
                )

            return vectors

        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Gemini embedding failed after "
                    f"{MAX_RETRIES} attempts: {exc}"
                ) from exc

            wait_seconds = min(2 ** attempt, 30)

            print(
                f"Embedding request failed: {exc}\n"
                f"Retrying in {wait_seconds} seconds "
                f"({attempt + 1}/{MAX_RETRIES})..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Embedding request failed.")


# --------------------------------------------------
# 5. Load PDFs and create chunk records
# --------------------------------------------------

def build_records() -> tuple[list[dict], int]:
    """Extract all PDF text and prepare chunk metadata."""

    if not DATA_DIR.exists():
        raise SystemExit(
            f"Data directory does not exist: {DATA_DIR}"
        )

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise SystemExit(
            f"No PDF files found directly inside: {DATA_DIR}"
        )

    records = []

    for pdf_path in pdf_files:
        print(f"\nExtracting: {pdf_path.name}")

        pages = extract_pdf_pages(pdf_path)

        if not pages:
            print(
                f"WARNING: No selectable text found in "
                f"{pdf_path.name}. It may be a scanned PDF."
            )
            continue

        for page_data in pages:
            chunks = split_text(page_data["text"])

            for chunk_number, chunk in enumerate(chunks, start=1):
                records.append({
                    "text": chunk,
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunk": chunk_number,
                })

        print(f"  Pages with text: {len(pages)}")

    if not records:
        raise SystemExit(
            "No text chunks were created. Check whether your "
            "PDFs contain selectable text or require OCR."
        )

    return records, len(pdf_files)


# --------------------------------------------------
# 6. Embed chunks and build FAISS index
# --------------------------------------------------

def main() -> None:
    if not GEMINI_API_KEY:
        raise SystemExit(
            "GEMINI_API_KEY is missing.\n"
            "Add your Google AI Studio API key to the .env file."
        )

    records, pdf_count = build_records()

    print(f"\nPDF files found: {pdf_count}")
    print(f"Text chunks created: {len(records)}")
    print(f"Embedding model: {EMBED_MODEL}")
    print("Connecting to Gemini...")

    client = genai.Client(api_key=GEMINI_API_KEY)
    all_vectors = []

    # Process chunks in batches to reduce API calls.
    # Process chunks in batches to reduce API calls.
    total = len(records)

    for start in range(0, total, BATCH_SIZE):
      batch_records = records[start : start + BATCH_SIZE]
      batch_texts = [item["text"] for item in batch_records]

      batch_number = start // BATCH_SIZE + 1
      total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

      print(
          f"Embedding batch {batch_number}/{total_batches}"
          f" ({len(batch_texts)} chunks)..."
      )

      vectors = embed_batch(client, batch_texts)
      all_vectors.extend(vectors)

      # 👉 ADD THIS PAUSE SO YOU DO NOT EXCEED THE PER-MINUTE LIMIT
      time.sleep(6.0)

    # Convert embeddings to the format expected by FAISS.
    matrix = np.asarray(all_vectors, dtype=np.float32)

    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise RuntimeError(
            "Embedding matrix shape does not match chunk count."
        )

    if not np.isfinite(matrix).all():
        raise RuntimeError(
            "Embedding matrix contains invalid numeric values."
        )

    # Normalize vectors so inner product gives cosine similarity.
    faiss.normalize_L2(matrix)

    # Create the searchable index.
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    # Save index and matching metadata.
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / "index.faiss"
    chunks_path = INDEX_DIR / "chunks.jsonl"

    faiss.write_index(index, str(index_path))

    with chunks_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print("\n" + "=" * 55)
    print("PDF INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 55)
    print(f"PDF files processed: {pdf_count}")
    print(f"Chunks indexed: {index.ntotal}")
    print(f"Embedding dimensions: {matrix.shape[1]}")
    print(f"FAISS index: {index_path}")
    print(f"Chunk metadata: {chunks_path}")
    print("\nYou can now start the RAG backend.")


if __name__ == "__main__":
    main()
