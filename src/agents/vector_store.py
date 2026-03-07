# import os
# from typing import List
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from src.models import LDU

# class VectorStoreManager:
#     def __init__(self, db_path: str = ".refinery/vectorstore"):
#         self.db_path = db_path
#         self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         self.vectorstore = None

#     def ingest_ldus(self, doc_id: str, ldus: List[LDU]):
#         texts = [ldu.content for ldu in ldus]
#         metadatas = [
#             {
#                 "doc_id": doc_id,
#                 "page_refs": ",".join(map(str, ldu.page_refs)),
#                 "chunk_type": ldu.chunk_type,
#                 "content_hash": ldu.content_hash,
#                 "parent_section": ldu.parent_section or ""
#             }
#             for ldu in ldus
#         ]
        
#         if os.path.exists(self.db_path):
#             self.vectorstore = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
#             self.vectorstore.add_texts(texts, metadatas=metadatas)
#         else:
#             self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            
#         self.vectorstore.save_local(self.db_path)

#     def search(self, query: str, k: int = 5):
#         if not self.vectorstore:
#             if os.path.exists(self.db_path):
#                 self.vectorstore = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
#             else:
#                 return []
        
#         return self.vectorstore.similarity_search(query, k=k)

import os
import json
from typing import List, Tuple, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.models import LDU


class VectorStoreManager:
    """
    One FAISS index per doc_id:
      .refinery/vectorstore/{doc_id}/
        - index.faiss
        - index.pkl
        - ingested_hashes.json
    """

    def __init__(self, base_path: str = ".refinery/vectorstore"):
        self.base_path = base_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore: Optional[FAISS] = None
        self.loaded_doc_id: Optional[str] = None

    # ---------- paths ----------
    def _doc_path(self, doc_id: str) -> str:
        return os.path.join(self.base_path, doc_id)

    def _hash_registry_path(self, doc_id: str) -> str:
        return os.path.join(self._doc_path(doc_id), "ingested_hashes.json")

    # ---------- registry ----------
    def _load_hash_registry(self, doc_id: str) -> set:
        path = self._hash_registry_path(doc_id)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        return set()

    def _save_hash_registry(self, doc_id: str, hashes: set) -> None:
        os.makedirs(self._doc_path(doc_id), exist_ok=True)
        with open(self._hash_registry_path(doc_id), "w", encoding="utf-8") as f:
            json.dump(sorted(hashes), f, ensure_ascii=False, indent=2)

    # ---------- load/create ----------
    def _load_or_none(self, doc_id: str) -> Optional[FAISS]:
        path = self._doc_path(doc_id)
        if os.path.exists(path):
            return FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True  # OK for local dev; keep folder trusted
            )
        return None

    def _ensure_loaded(self, doc_id: str) -> None:
        if self.vectorstore is not None and self.loaded_doc_id == doc_id:
            return
        self.vectorstore = self._load_or_none(doc_id)
        self.loaded_doc_id = doc_id

    # ---------- ingest ----------
    def ingest_ldus(self, doc_id: str, ldus: List[LDU]) -> None:
        self._ensure_loaded(doc_id)

        seen = self._load_hash_registry(doc_id)

        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        new_hashes: set = set()

        for ldu in ldus:
            h = ldu.content_hash
            if h in seen:
                continue

            texts.append(ldu.content)

            page_min = min(ldu.page_refs) if ldu.page_refs else -1
            page_max = max(ldu.page_refs) if ldu.page_refs else -1

            meta = {
                "doc_id": doc_id,
                "content_hash": h,
                "chunk_type": ldu.chunk_type,
                "parent_section": ldu.parent_section or "",
                "page_refs": ldu.page_refs,     # list is fine in metadata
                "page_min": page_min,
                "page_max": page_max,
            }

            # Optional: store bbox if you want it directly in search results
            if ldu.bounding_box is not None:
                meta["bbox"] = ldu.bounding_box.model_dump()

            metadatas.append(meta)
            new_hashes.add(h)

        if not texts:
            return

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        else:
            self.vectorstore.add_texts(texts, metadatas=metadatas)

        out_path = self._doc_path(doc_id)
        os.makedirs(out_path, exist_ok=True)
        self.vectorstore.save_local(out_path)

        seen |= new_hashes
        self._save_hash_registry(doc_id, seen)

    # ---------- search ----------
    def search(
        self,
        doc_id: str,
        query: str,
        k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        self._ensure_loaded(doc_id)
        if self.vectorstore is None:
            return []

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [(d.page_content, float(score), d.metadata) for d, score in results]

    def search_in_page_ranges(
        self,
        doc_id: str,
        query: str,
        page_ranges: List[Tuple[int, int]],
        k: int = 5,
        fetch_k: int = 50
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        PageIndex-assisted retrieval:
        - FAISS can't filter by metadata during search.
        - So fetch more (fetch_k), filter by page range overlap, return top k.
        """
        self._ensure_loaded(doc_id)
        if self.vectorstore is None:
            return []

        raw = self.vectorstore.similarity_search_with_score(query, k=fetch_k)

        def overlaps(meta: Dict[str, Any]) -> bool:
            pmin = int(meta.get("page_min", -1))
            pmax = int(meta.get("page_max", -1))
            for a, b in page_ranges:
                if pmax >= a and pmin <= b:
                    return True
            return False

        filtered: List[Tuple[str, float, Dict[str, Any]]] = []
        for d, score in raw:
            if overlaps(d.metadata):
                filtered.append((d.page_content, float(score), d.metadata))
                if len(filtered) >= k:
                    break

        return filtered