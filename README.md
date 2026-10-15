# Tamil Nadu Agriculture Schemes Assistant — RAG

A PDF-grounded RAG chatbot with a FastAPI backend and Streamlit frontend. It retrieves passages from PDFs in `Data/`, answers in English or Tamil, and returns PDF/page sources.

## Project layout

```text
W2_Buildathon/
├── Data/                  # Put the scheme PDFs here
├── faiss_index/            # Created by ingest.py
├── ingest.py
├── rag.py
├── guardrails.py
├── api.py
├── app.py
├── eval_questions.csv
├── eval_deepeval.py
├── requirements.txt
├── .env.example
└── chats.json              # Created at runtime
```

## Windows setup

Open PowerShell in `D:\SocialEagle\Program\W2_Buildathon`.

Use Python 3.12 (recommended for compatibility):

```powershell
py -3.12 -m venv venv312
.\venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Add your OpenAI API key to `.env`. The default embedding and chat models are `text-embedding-3-small` and `gpt-4o-mini`. This implementation uses OpenAI API calls, so an API key and usage budget are required.

## Add PDFs and build the index

Place the three PDFs (or other official agriculture scheme PDFs) in `Data/`, then run:

```powershell
python ingest.py
```

This creates `faiss_index\index.faiss` and `faiss_index\chunks.jsonl`. Run ingestion again only after adding/changing PDFs.

PDF text extraction uses PyMuPDF. Tables that are represented as text are included in the extracted text. Scanned pages get an optional local OCR attempt if `pytesseract` and Pillow are installed and the Tesseract executable is configured. Complex charts and image-only tables may need manual verification; this baseline does not call a vision model for every page.

## Run the app

Terminal 1 — backend:

```powershell
.\venv312\Scripts\Activate.ps1
uvicorn api:app --reload --port 8000
```

Terminal 2 — frontend:

```powershell
.\venv312\Scripts\Activate.ps1
streamlit run app.py
```

Open the Streamlit URL shown in the terminal. API docs: `http://localhost:8000/docs`; health: `http://localhost:8000/health`.

## API example

```json
{
  "question": "What subsidies are available for small farmers?",
  "history": []
}
```

Use `POST http://localhost:8000/chat` for a complete response or `POST /chat/stream` for newline-delimited JSON events.

## Guardrails and limitations

Rule-based checks reject empty/overlong prompts and a few common injection patterns; output checks block empty output and key-like strings. This version does **not** load NVIDIA NeMo Guardrails by default, and `/health` reports this accurately. Keyword rules are not a complete security boundary. Answers are limited to retrieved PDF text, but always verify important eligibility, dates, and amounts against the cited source page and local agriculture office.

Chat history is stored in `chats.json` on the local machine. Do not use this shared-file approach for multi-user production without adding user authentication and per-user storage.

## Evaluation

```powershell
python eval_deepeval.py
```

Set `N=1` to test one question. Evaluation uses model calls and may incur API charges. DeepEval API compatibility can vary by version.

## Troubleshooting

- `OPENAI_API_KEY is missing`: edit `.env`; ensure it is in the project root.
- `FAISS index not found`: run `python ingest.py` after adding PDFs.
- `No PDF files found`: confirm PDFs are directly inside `Data`, not a nested folder.
- PowerShell activation blocked: run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then activate again.
- Tamil text extraction: check that the PDF contains a Unicode text layer. For scanned PDFs, install Tesseract OCR and optionally `pip install pytesseract pillow`.
