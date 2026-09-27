from pathlib import Path

import pandas as pd
import streamlit as st

from recommender import ContentBasedBookRecommender, REQUIRED_COLUMNS

st.set_page_config(page_title="个性化图书推荐系统", page_icon="📚", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "books.csv"


@st.cache_resource
def load_default_model() -> ContentBasedBookRecommender:
    return ContentBasedBookRecommender.from_csv(DATA_PATH)


st.title("📚 基于内容相似度的个性化图书推荐系统")
st.caption("Python + Pandas + Scikit-learn + Streamlit | TF-IDF 与余弦相似度")

with st.sidebar:
    st.header("数据设置")
    uploaded = st.file_uploader("可选：上传自定义图书 CSV", type=["csv"])
    st.markdown("CSV 必需字段：`title, author, genre, description, rating`")

try:
    if uploaded is not None:
        custom_df = pd.read_csv(uploaded)
        missing = REQUIRED_COLUMNS - set(custom_df.columns)
        if missing:
            st.error("缺少字段：" + ", ".join(sorted(missing)))
            st.stop()
        model = ContentBasedBookRecommender(custom_df)
        source_label = "自定义数据"
    else:
        model = load_default_model()
        source_label = "内置示例数据"
except Exception as exc:
    st.error(f"数据加载失败：{exc}")
    st.stop()

st.success(f"已加载 {len(model.books)} 本图书（{source_label}）")

recommend_tab, search_tab, info_tab = st.tabs(["智能推荐", "图书检索", "系统说明"])

with recommend_tab:
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_title = st.selectbox("选择一本你喜欢的书", model.titles)
    with col2:
        top_n = st.slider("推荐数量", min_value=3, max_value=10, value=5)

    selected_genre = model.books.loc[model.books["title"] == selected_title, "genre"].iloc[0]
    st.write(f"当前图书类型：**{selected_genre}**")

    if st.button("生成推荐", type="primary", use_container_width=True):
        result = model.recommend(selected_title, top_n=top_n)
        st.subheader("推荐结果")
        for rank, row in result.reset_index(drop=True).iterrows():
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"### {rank + 1}. {row['title']}")
                    st.write(f"作者：{row['author']} | 类型：{row['genre']}")
                    st.write(row["description"])
                with right:
                    st.metric("相似度", f"{row['similarity']:.3f}")
                    st.metric("评分", f"{row['rating']:.1f}")

        st.dataframe(
            result[["title", "author", "genre", "rating", "similarity"]],
            hide_index=True,
            use_container_width=True,
        )

with search_tab:
    keyword = st.text_input("按书名、作者或简介关键词搜索")
    genre_options = sorted(model.books["genre"].unique().tolist())
    chosen_genres = st.multiselect("按类型筛选", genre_options)
    result = model.search(keyword, chosen_genres)
    st.dataframe(result, hide_index=True, use_container_width=True)

with info_tab:
    st.markdown(
        """
        ### 工作流程
        1. 读取图书元数据并清洗空值、重复记录；
        2. 将书名、作者、类型和简介组合为文本特征；
        3. 使用 TF-IDF 将文本转换为向量；
        4. 使用余弦相似度计算图书之间的内容相似程度；
        5. 排除用户当前选择的图书后，返回相似度最高的 Top-N 图书。

        ### 设计特点
        - 完全离线运行，不依赖付费 API；
        - 模型轻量，普通电脑即可运行；
        - 支持替换为自定义 CSV 数据集；
        - 推荐结果包含相似度与评分，便于解释推荐原因。
        """
    )
