from pathlib import Path

import pandas as pd
import pytest

from recommender import ContentBasedBookRecommender

DATA_PATH = Path(__file__).parents[1] / "data" / "books.csv"


def test_load_and_titles():
    model = ContentBasedBookRecommender.from_csv(DATA_PATH)
    assert len(model.books) >= 20
    assert "1984" in model.titles


def test_recommend_returns_requested_rows():
    model = ContentBasedBookRecommender.from_csv(DATA_PATH)
    result = model.recommend("1984", top_n=5)
    assert len(result) == 5
    assert "1984" not in result["title"].tolist()
    assert result["similarity"].between(0, 1).all()


def test_unknown_title_raises_key_error():
    model = ContentBasedBookRecommender.from_csv(DATA_PATH)
    with pytest.raises(KeyError):
        model.recommend("A Book That Does Not Exist")


def test_missing_column_raises_value_error():
    df = pd.DataFrame({"title": ["A"], "author": ["B"]})
    with pytest.raises(ValueError):
        ContentBasedBookRecommender(df)


def test_search_keyword():
    model = ContentBasedBookRecommender.from_csv(DATA_PATH)
    result = model.search("dystopian")
    assert len(result) > 0
