import json
import os
from typing import List

from pdf_loader import load_multiple_pdfs
from chunker import chunk_pages, merge_nearby_chunks
from rag import RAG


def main():
    print("Local Ollama PDF Chat (RAG)")
    persist_dir = os.path.join(os.path.abspath('.'), 'chroma_db')
    rag = RAG(persist_dir)
    memory_file = "memory.json"
    history = []
    if os.path.exists(memory_file):
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                history = json.load(f)
                # Ensure history is a list (file may contain a single dict from older runs)
                if not isinstance(history, list):
                    if isinstance(history, dict):
                        history = [history]
                    else:
                        history = []
        except Exception:
            history = []

    while True:
        print("\nOptions: 1) Load PDF 2) Ask question 3) List docs 4) Exit")
        choice = input("Select: ")
        if choice.strip() == "1":
            path = input("Path to PDF (comma-separate for multiple): ")
            paths = [p.strip() for p in path.split(',') if p.strip()]
            pages = load_multiple_pdfs(paths)
            chunks = chunk_pages(pages)
            chunks = merge_nearby_chunks(chunks)
            rag.ingest_chunks(chunks)
            print(f"Ingested {len(chunks)} chunks from: {paths}")
        elif choice.strip() == "2":
            q = input("Question: ")
            res = rag.answer_question(q)
            print("\nAnswer:\n", res["answer"])
            history.append({"question": q, "answer": res["answer"]})
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        elif choice.strip() == "3":
            # show basic info about stored docs
            try:
                count = rag.vs.count()
                print(f"Stored vectors (approx): {count}")
            except Exception as e:
                print("Failed to read store:", e)
        else:
            break


if __name__ == "__main__":
    main()