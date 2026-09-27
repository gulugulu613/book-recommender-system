# 基于内容相似度的个性化图书推荐系统

## 1. 项目简介
本项目使用 Python、Pandas、Scikit-learn 和 Streamlit，实现一个轻量级、可解释、离线运行的图书推荐系统。核心算法为 TF-IDF 文本向量化与余弦相似度。

## 2. 环境要求
- Python 3.10+
- Windows / macOS / Linux

## 3. 安装依赖
```bash
pip install -r requirements.txt
```

## 4. 运行系统
```bash
streamlit run app.py
```
浏览器通常会自动打开 `http://localhost:8501`。

## 5. 运行测试
```bash
pytest -q
```

## 6. 数据格式
默认数据位于 `data/books.csv`。也可以在网页侧边栏上传自己的 CSV，必须包含：
- title
- author
- genre
- description
- rating

## 7. 项目结构
```text
book_recommender_project/
├─ app.py
├─ recommender.py
├─ requirements.txt
├─ README.md
├─ data/
│  └─ books.csv
└─ tests/
   └─ test_recommender.py
```
