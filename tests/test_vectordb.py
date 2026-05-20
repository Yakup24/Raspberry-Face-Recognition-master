from pathlib import Path

import numpy as np
import pytest

from raspberry_face_recognition.vectordb import FaceVectorDB, VectorDBError


class FakeIndex:
    def __init__(self, dim):
        self.d = dim
        self.vectors = np.empty((0, dim), dtype=np.float32)

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, matrix):
        self.vectors = np.vstack([self.vectors, matrix.astype(np.float32)])

    def search(self, matrix, count):
        if self.ntotal == 0:
            return np.array([[float("inf")]], dtype=np.float32), np.array([[-1]], dtype=np.int64)
        distances = np.sum((self.vectors - matrix[0]) ** 2, axis=1)
        index = int(np.argmin(distances))
        return np.array([[float(distances[index])]], dtype=np.float32), np.array([[index]], dtype=np.int64)


class FakeFaiss:
    @staticmethod
    def IndexFlatL2(dim):
        return FakeIndex(dim)

    @staticmethod
    def normalize_L2(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrix[:] = matrix / norms

    @staticmethod
    def write_index(index, path):
        Path(path).write_text("fake-index", encoding="utf-8")

    @staticmethod
    def read_index(path):
        if not Path(path).exists():
            raise FileNotFoundError(path)
        return FakeIndex(512)


def test_vector_db_adds_and_finds_embedding(tmp_path):
    db = FaceVectorDB(
        tmp_path / "faiss.index",
        tmp_path / "labels.json",
        faiss_module=FakeFaiss,
        numpy_module=np,
    )

    db.add_face(np.ones((1, 512), dtype=np.float32), "demo-user")
    result = db.search_face(np.ones((1, 512), dtype=np.float32), threshold=0.01)

    assert db.count == 1
    assert result.matched is True
    assert result.label == "demo-user"
    assert (tmp_path / "labels.json").exists()


def test_vector_db_returns_unknown_when_empty(tmp_path):
    db = FaceVectorDB(
        tmp_path / "faiss.index",
        tmp_path / "labels.json",
        faiss_module=FakeFaiss,
        numpy_module=np,
    )

    result = db.search_face(np.zeros((1, 512), dtype=np.float32), threshold=0.8)

    assert result.matched is False
    assert result.label == "unknown"


def test_vector_db_rejects_wrong_embedding_shape(tmp_path):
    db = FaceVectorDB(
        tmp_path / "faiss.index",
        tmp_path / "labels.json",
        faiss_module=FakeFaiss,
        numpy_module=np,
    )

    with pytest.raises(VectorDBError):
        db.add_face(np.zeros((1, 128), dtype=np.float32), "demo-user")
