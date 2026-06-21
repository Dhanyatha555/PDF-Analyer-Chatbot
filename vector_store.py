from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
import os
import hashlib


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "pdf_chunks"):
        os.makedirs(persist_directory, exist_ok=True)
        # Try a modern persistent configuration first (duckdb+parquet).
        # If the user's chromadb installation uses the legacy config, fall back to a default client.
        self.collection = None
        try:
            settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
            self.client = chromadb.Client(settings)
        except Exception:
            # Fallback: try default client (may be in-memory depending on chromadb version)
            try:
                self.client = chromadb.Client()
            except Exception as e:
                raise RuntimeError("Failed to initialize Chroma client: " + str(e))

        # Robust collection acquisition across Chroma versions
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            try:
                self.collection = self.client.create_collection(name=collection_name)
            except Exception:
                # Some versions expose get_or_create_collection
                try:
                    self.collection = self.client.get_or_create_collection(name=collection_name)
                except Exception as e:
                    raise RuntimeError("Failed to get or create Chroma collection: " + str(e))

    def add_chunks(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], documents: List[str]):
        """Add documents with embeddings to the Chroma collection."""
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        # Persist if supported by the client implementation
        try:
            self.client.persist()
        except Exception:
            pass

    def query(self, embedding: List[float], top_k: int = 5) -> Dict:
        """Query the collection and return results with distances."""
        # Support different chromadb versions which expect different kwarg names.
        include = ['metadatas', 'documents', 'distances', 'embeddings']
        try:
            # modern API: query_embeddings
            res = self.collection.query(query_embeddings=[embedding], n_results=top_k, include=include)
        except TypeError:
            try:
                # older/other API: embedding
                res = self.collection.query(embedding=embedding, n_results=top_k, include=include)
            except Exception as e:
                raise RuntimeError("Chroma collection.query failed: " + str(e))
        except Exception as e:
            raise RuntimeError("Chroma collection.query failed: " + str(e))

        # Normalize result and synthesize ids if the Chroma version doesn't return them.
        normalized = {}
        for key in ['documents', 'metadatas', 'distances', 'embeddings', 'ids', 'uris', 'data']:
            normalized[key] = res.get(key)

        # If ids missing, synthesize stable ids from document text
        if not normalized.get('ids'):
            docs = normalized.get('documents') or []
            if docs and isinstance(docs, list) and len(docs) > 0:
                first_list = docs[0]
                gen_ids = []
                for i, doc in enumerate(first_list):
                    h = hashlib.sha1(doc.encode('utf-8')).hexdigest()[:8]
                    gen_ids.append(f"gen-{h}-{i}")
                normalized['ids'] = [gen_ids]
            else:
                normalized['ids'] = [[]]

        return normalized

    def count(self) -> int:
        return self.collection.count()
