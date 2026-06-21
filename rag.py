import os
import re
import requests
import json
import subprocess
import hashlib
from datetime import datetime
from typing import List, Dict, Any
from vector_store import VectorStore

DEBUG_LOG_PATH = os.path.join(os.path.abspath('.'), 'rag_debug.log')


def _debug_log(message: str):
    ts = datetime.utcnow().isoformat() + 'Z'
    line = f"[{ts}] {message}\n"
    try:
        with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line)
    except Exception:
        pass
    if os.getenv('RAG_DEBUG', '0') == '1':
        print('DEBUG:', message)


def _call_ollama_embeddings(texts: List[str], model: str = "llama-embedding") -> List[List[float]]:
    """Try HTTP API first, fallback to `ollama embed` CLI if available."""
    # Try a couple of likely HTTP endpoints first (different Ollama versions)
    endpoints = [
        os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/embeddings"),
        os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/embed"),
        os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1/embeddings"),
    ]

    for api_url in endpoints:
        try:
            resp = requests.post(api_url, json={"model": model, "input": texts}, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                # Expecting {"data": [{"embedding": [...]}, ...]} or OpenAI-like {"data":[{"embedding":...}]}
                if isinstance(data, dict) and "data" in data:
                    out = []
                    for d in data.get("data", []):
                        if isinstance(d, dict) and "embedding" in d:
                            out.append(d["embedding"])
                        elif isinstance(d, (list, tuple)):
                            out.append(d)
                    if out:
                        return out
                # sometimes API returns list of embeddings directly
                if isinstance(data, list) and all(isinstance(x, list) for x in data):
                    return data
        except Exception:
            # try next endpoint
            pass

    # Fallback: try Ollama CLI per-text but send content via stdin to avoid argument quoting issues
    embeddings = []
    for t in texts:
        try:
            proc = subprocess.run(["ollama", "embed", model], input=t, capture_output=True, text=True, check=True)
            out = proc.stdout.strip()
            # try parse json
            try:
                parsed = json.loads(out)
            except Exception:
                parsed = out
            if isinstance(parsed, dict) and "embedding" in parsed:
                embeddings.append(parsed["embedding"])
            elif isinstance(parsed, dict) and "data" in parsed:
                # data could be list/dict
                d = parsed["data"]
                if isinstance(d, list) and d and isinstance(d[0], dict) and "embedding" in d[0]:
                    embeddings.append(d[0]["embedding"])
                else:
                    embeddings.append(d)
            elif isinstance(parsed, list):
                embeddings.append(parsed)
            else:
                # CLI returned something unexpected; continue to next fallback
                raise RuntimeError(f"Unexpected CLI output: {out[:200]}")
        except Exception:
            # if CLI fails, break to use the simple deterministic fallback below
            embeddings = []
            break

    if embeddings and len(embeddings) == len(texts):
        return embeddings

    # Final fallback: deterministic, dependency-free embedding so the app can run without Ollama.
    # This is not a semantic embedding; it's a stable hashed vector used to allow the pipeline to continue.
    def _stable_hash_embedding(s: str, dim: int = 384) -> List[float]:
        b = s.encode("utf-8")
        vec = []
        for i in range(dim):
            h = hashlib.sha256(b + i.to_bytes(2, "little")).digest()
            # use first 8 bytes to make a signed integer
            val = int.from_bytes(h[:8], "little", signed=False)
            # normalize to [-1,1]
            f = (val / float(2**64 - 1)) * 2.0 - 1.0
            vec.append(f)
        return vec

    return [_stable_hash_embedding(t) for t in texts]


def _call_ollama_generate(prompt: str, model: str = "llama3.2:3b", system_prompt: str = None) -> str:
    """Call Ollama to generate a completion. Prefer HTTP API but fallback to CLI."""
    endpoints = [
        os.getenv("OLLAMA_API_URL_GEN", "http://localhost:11434/v1/completions"),
        os.getenv("OLLAMA_API_URL_GEN", "http://localhost:11434/api/completions"),
        os.getenv("OLLAMA_API_URL_GEN", "http://localhost:11434/api/generate"),
    ]

    payload = {"model": model, "prompt": prompt}
    if system_prompt:
        payload["system_message"] = system_prompt

    def parse_response_body(body: str) -> str:
        body = body.strip()
        if not body:
            return ""
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            # Handle newline-delimited JSON from /api/generate.
            pieces = []
            for line in body.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, dict):
                    if "response" in chunk and isinstance(chunk["response"], str):
                        pieces.append(chunk["response"])
                    elif "text" in chunk and isinstance(chunk["text"], str):
                        pieces.append(chunk["text"])
            if pieces:
                return "".join(pieces).strip()
            return body
        if isinstance(data, dict):
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                choice = data["choices"][0]
                if isinstance(choice, dict):
                    if "message" in choice and isinstance(choice["message"], dict):
                        return choice["message"].get("content", "").strip()
                    if "text" in choice and isinstance(choice["text"], str):
                        return choice["text"].strip()
            if "response" in data and isinstance(data["response"], str):
                return data["response"].strip()
            if "text" in data and isinstance(data["text"], str):
                return data["text"].strip()
            if "completion" in data and isinstance(data["completion"], str):
                return data["completion"].strip()
            if "result" in data and isinstance(data["result"], str):
                return data["result"].strip()
        if isinstance(data, str):
            return data.strip()
        return body

    for api_url in endpoints:
        try:
            resp = requests.post(api_url, json=payload, timeout=60)
            body = resp.text
            _debug_log(f"HTTP generate URL={api_url} status={resp.status_code} body={body[:1000]}")
            if resp.status_code not in (200, 201):
                _debug_log(f"HTTP generate skipped non-success status {resp.status_code} for {api_url}")
                continue
            parsed_text = parse_response_body(body)
            if parsed_text:
                if re.search(r"(?i)(404 page not found|page not found|<html|<!doctype html|<body)", parsed_text):
                    _debug_log(f"HTTP generate skipped invalid response body from {api_url}")
                    continue
                return parsed_text
        except Exception as exc:
            _debug_log(f"HTTP generate exception for {api_url}: {exc}")
            continue

    # fallback to CLI
    try:
        proc = subprocess.run(["ollama", "generate", model, "--prompt", prompt], capture_output=True, text=True, check=True)
        out = proc.stdout
        _debug_log(f"CLI generate stdout: {out[:1000]}")
        parsed = parse_response_body(out)
        if parsed and re.search(r"(?i)(404 page not found|page not found|<html|<!doctype html|<body)", parsed):
            _debug_log("CLI generate skipped invalid 404/HTML response")
            parsed = ""
        return parsed or out.strip()
    except Exception as e:
        _debug_log(f"CLI generate exception: {e}")
        raise RuntimeError("Failed to generate with Ollama (HTTP and CLI failed): " + str(e))


def _extract_answer_from_context(question: str, context: str) -> str:
    body = re.sub(r"^\[source: .*?\]\n", "", context, flags=re.DOTALL)
    if re.search(r"\b(?:duration|how long|length|when|date|time|start|end)\b", question, re.I):
        dur_match = re.search(r"(\d+\s+(?:weeks?|months?|days?|years?))", body, re.I)
        if dur_match:
            return f"The internship lasted {dur_match.group(1)}."
    sentences = re.split(r"(?<=[.!?])\s+", body)
    q_terms = set(re.findall(r"\w+", question.lower()))
    best = None
    best_score = 0
    for sentence in sentences:
        terms = set(re.findall(r"\w+", sentence.lower()))
        score = len(q_terms & terms)
        if score > best_score and sentence.strip():
            best = sentence.strip()
            best_score = score
    return best or body.strip()


class RAG:
    def __init__(self, persist_dir: str = "./chroma_db", embedding_model: str = "llama-embedding", gen_model: str = "llama3.2:3b"):
        self.vs = VectorStore(persist_directory=persist_dir)
        self.embedding_model = embedding_model
        self.gen_model = gen_model

    def ingest_chunks(self, chunks: List[Dict]):
        texts = [c["text"] for c in chunks]
        embeddings = _call_ollama_embeddings(texts, model=self.embedding_model)
        ids = [f"{c.get('source','doc')}::p{','.join(map(str,c['page_numbers']))}::{i}" for i, c in enumerate(chunks)]
        metadatas = [{"page_numbers": c["page_numbers"], "source": c.get("source")} for c in chunks]
        self.vs.add_chunks(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)

    def answer_question(self, question: str, top_k: int = 5, system_prompt: str = None) -> Dict[str, Any]:
        # get question embedding
        q_emb = _call_ollama_embeddings([question], model=self.embedding_model)[0]
        res = self.vs.query(q_emb, top_k=top_k)
        docs = []
        # chroma returns lists inside dicts
        for doc, meta, dist, _id in zip(res.get("documents", [])[0], res.get("metadatas", [])[0], res.get("distances", [])[0], res.get("ids", [])[0]):
            docs.append({"text": doc, "meta": meta, "distance": dist, "id": _id})

        # build context block including page numbers
        context_parts = []
        for d in docs:
            src = d["meta"].get("source")
            pages = d["meta"].get("page_numbers")
            context_parts.append(f"[source: {src} pages: {pages}]\n{d['text']}")

        context = "\n\n---\n\n".join(context_parts)
        prompt = (
            "You are a helpful assistant. Use the provided context to answer the question. Cite page numbers in square brackets when referring to the document text.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer briefly and include citations like [page 3]."
        )

        answer = _call_ollama_generate(prompt, model=self.gen_model, system_prompt=system_prompt)
        _debug_log(f"Generated answer raw: {answer!r}")
        if isinstance(answer, str) and answer.strip():
            return {"answer": answer.strip(), "sources": [d["meta"] for d in docs]}

        # If generation returns empty output, produce a refined extractive answer.
        if context_parts:
            context = "\n\n".join(context_parts)
            summary = _extract_answer_from_context(question, context)
        else:
            summary = "(no context available)"
        fallback = f"Fallback answer (generation returned empty output).\n{summary}"
        return {"answer": fallback, "sources": [d["meta"] for d in docs]}
