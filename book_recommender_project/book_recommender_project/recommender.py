from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_COLUMNS = {"title", "author", "genre", "description", "rating"}


@dataclass
class RecommendationResult:
    title: str
    author: str
    genre: str
    rating: float
    similarity: float
    description: str


class ContentBasedBookRecommender:
    """Lightweight TF-IDF + cosine similarity recommender.

    The model is intentionally simple and fully offline so that it can be
    demonstrated in a classroom environment without a paid API or GPU.
    """

    def __init__(self, books: pd.DataFrame):
        self.books = self._prepare_data(books)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.books["content"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        self._title_to_index = {
            title.lower().strip(): idx for idx, title in enumerate(self.books["title"])
        }

    @staticmethod
    def _prepare_data(books: pd.DataFrame) -> pd.DataFrame:
        missing = REQUIRED_COLUMNS - set(books.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        data = books.copy()
        for col in ["title", "author", "genre", "description"]:
            data[col] = data[col].fillna("").astype(str).str.strip()
        data["rating"] = pd.to_numeric(data["rating"], errors="coerce").fillna(0.0)
        data = data[data["title"] != ""].drop_duplicates(subset=["title", "author"]).reset_index(drop=True)

        # Repeat genre and author fields to give them slightly more influence than
        # free-form description text while preserving model simplicity.
        data["content"] = (
            (data["genre"] + " ") * 3
            + (data["author"] + " ") * 2
            + data["title"]
            + " "
            + data["description"]
        ).str.lower()
        return data

    @classmethod
    def from_csv(cls, path: str | Path) -> "ContentBasedBookRecommender":
        return cls(pd.read_csv(path))

    @property
    def titles(self) -> list[str]:
        return sorted(self.books["title"].tolist())

    def recommend(self, title: str, top_n: int = 5, genre_filter: str | None = None) -> pd.DataFrame:
        if top_n <= 0:
            raise ValueError("top_n must be greater than 0")

        idx = self._title_to_index.get(title.lower().strip())
        if idx is None:
            raise KeyError(f"Book not found: {title}")

        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        rows = []
        for book_idx, score in scores:
            if book_idx == idx:
                continue
            row = self.books.iloc[book_idx]
            if genre_filter and genre_filter != "All" and row["genre"] != genre_filter:
                continue
            rows.append(
                {
                    "title": row["title"],
                    "author": row["author"],
                    "genre": row["genre"],
                    "rating": float(row["rating"]),
                    "similarity": float(score),
                    "description": row["description"],
                }
            )
            if len(rows) >= top_n:
                break

        return pd.DataFrame(rows)

    def search(self, keyword: str, genres: Iterable[str] | None = None) -> pd.DataFrame:
        keyword = keyword.strip().lower()
        data = self.books.copy()
        if keyword:
            mask = (
                data["title"].str.lower().str.contains(keyword, regex=False)
                | data["author"].str.lower().str.contains(keyword, regex=False)
                | data["description"].str.lower().str.contains(keyword, regex=False)
                | data["genre"].str.lower().str.contains(keyword, regex=False)
            )
            data = data[mask]
        if genres:
            genres = list(genres)
            data = data[data["genre"].isin(genres)]
        return data[["title", "author", "genre", "rating", "description"]].sort_values(
            ["rating", "title"], ascending=[False, True]
        )
