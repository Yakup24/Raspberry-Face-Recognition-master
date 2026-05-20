"""FAISS-backed face embedding index.

The database stores embeddings and labels, not raw face images. Embeddings are
still biometric data and should be treated as sensitive local artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional, Union


class VectorDBError(RuntimeError):
    """Raised when the vector database cannot be loaded or queried."""


@dataclass(frozen=True)
class VectorSearchResult:
    label: str
    distance: float
    matched: bool
    index_id: int = -1


def require_faiss() -> Any:
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is not installed. Install the deep runtime with: "
            'python -m pip install -e ".[deep]"'
        ) from exc
    return faiss


def _as_embedding_matrix(np: Any, embedding: Any, dim: int) -> Any:
    matrix = np.asarray(embedding, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != dim:
        raise VectorDBError(f"Embedding must have shape (n, {dim}); got {matrix.shape}.")
    return matrix


class FaceVectorDB:
    """Small FAISS index with JSON label metadata."""

    def __init__(
        self,
        index_path: Union[str, Path] = "data/embeddings/faiss.index",
        labels_path: Union[str, Path] = "data/embeddings/labels.json",
        dim: int = 512,
        faiss_module: Optional[Any] = None,
        numpy_module: Optional[Any] = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.labels_path = Path(labels_path)
        self.dim = dim
        self.faiss = faiss_module or require_faiss()
        if numpy_module is None:
            import numpy as np  # type: ignore[import-not-found]

            self.np = np
        else:
            self.np = numpy_module
        self.index = self.faiss.IndexFlatL2(self.dim)
        self.labels: dict[int, str] = {}
        self.load_db()

    @property
    def count(self) -> int:
        return int(getattr(self.index, "ntotal", len(self.labels)))

    def add_face(self, embedding: Any, person_name: str) -> None:
        label = str(person_name).strip()
        if not label:
            raise VectorDBError("person_name must not be empty.")
        matrix = _as_embedding_matrix(self.np, embedding, self.dim)
        self.faiss.normalize_L2(matrix)
        start_id = self.count
        self.index.add(matrix)
        for offset in range(matrix.shape[0]):
            self.labels[start_id + offset] = label
        self.save_db()

    def search_face(self, embedding: Any, threshold: float = 0.8, unknown_label: str = "unknown") -> VectorSearchResult:
        if self.count == 0:
            return VectorSearchResult(unknown_label, float("inf"), False)
        matrix = _as_embedding_matrix(self.np, embedding, self.dim)
        self.faiss.normalize_L2(matrix)
        distances, indices = self.index.search(matrix, 1)
        distance = float(distances[0][0])
        index_id = int(indices[0][0])
        if index_id != -1 and distance <= threshold:
            return VectorSearchResult(self.labels.get(index_id, unknown_label), distance, True, index_id)
        return VectorSearchResult(unknown_label, distance, False, index_id)

    def save_db(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(self.index_path))
        payload = {
            "embedding_dim": self.dim,
            "labels": {str(index): label for index, label in sorted(self.labels.items())},
        }
        self.labels_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def load_db(self) -> None:
        if self.index_path.exists():
            self.index = self.faiss.read_index(str(self.index_path))
            if int(getattr(self.index, "d", self.dim)) != self.dim:
                raise VectorDBError(f"FAISS index dimension does not match expected dim={self.dim}.")
        if self.labels_path.exists():
            payload = json.loads(self.labels_path.read_text(encoding="utf-8"))
            if int(payload.get("embedding_dim", self.dim)) != self.dim:
                raise VectorDBError(f"Label metadata dimension does not match expected dim={self.dim}.")
            self.labels = {int(index): str(label) for index, label in payload.get("labels", {}).items()}
