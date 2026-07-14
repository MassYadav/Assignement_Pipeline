import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

logger = logging.getLogger(__name__)

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Initializing Semantic Search with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents: list[dict] = []

    def index_clauses(self, contract_id: str, extraction: dict):
        """Indexes the clauses of a contract."""
        clauses = [
            ("summary", extraction.get("summary", "")),
            ("termination_clause", extraction.get("termination_clause", "")),
            ("confidentiality_clause", extraction.get("confidentiality_clause", "")),
            ("liability_clause", extraction.get("liability_clause", ""))
        ]

        for clause_type, text in clauses:
            if text and text.lower() != "not found":
                embedding = self.model.encode([text])[0]
                self.index.add(np.array([embedding], dtype=np.float32))
                self.documents.append({
                    "contract_id": contract_id,
                    "clause_type": clause_type,
                    "text": text
                })
        logger.info(f"Indexed clauses for {contract_id}")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Searches the indexed clauses for the given query."""
        if self.index.ntotal == 0:
            logger.warning("Search index is empty.")
            return []

        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "contract_id": doc["contract_id"],
                    "clause_type": doc["clause_type"],
                    "text": doc["text"],
                    "score": float(dist)
                })
        return results
