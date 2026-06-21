# Local PDF Question Answering (RAG) with Ollama + ChromaDB

This project turns a local Ollama chatbot into a PDF Question Answering system using Retrieval-Augmented Generation (RAG).

Setup
- Install Ollama and the models you want (embedding model and `llama3.2:3b`).
- Create a Python virtualenv and install requirements:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Usage
- Run the chatbot:

```bash
python chatbot.py
```

- Options:
  - Load PDF(s): provide one or more comma-separated paths. The app extracts text with PyMuPDF, chunks it (~500 words with overlap), generates embeddings with Ollama, and stores them in ChromaDB.
  - Ask question: queries the vector store, assembles context (with page numbers), and sends it to `llama3.2:3b` to answer. The assistant is instructed to cite page numbers.

Notes
- The code prefers Ollama HTTP API at `http://localhost:11434`. If you don't use the HTTP API, the code will try the `ollama` CLI as a fallback for `embed` and `generate` commands.
- ChromaDB data is stored in `./chroma_db` by default.
- This is a minimal, beginner-friendly implementation. Adjust chunk sizes, models, and prompts as needed.
