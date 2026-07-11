import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CSS ----------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(-45deg,#141E30,#243B55,#0F2027,#2C5364);
background-size:400% 400%;
animation:gradient 15s ease infinite;
color:white;
}

@keyframes gradient{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

.big-title{
text-align:center;
font-size:55px;
font-weight:bold;
color:white;
margin-top:20px;
margin-bottom:20px;
}

.subtitle{
text-align:center;
font-size:20px;
color:#dddddd;
margin-bottom:30px;
}

.movie-card{
background:rgba(255,255,255,0.12);
padding:18px;
border-radius:20px;
backdrop-filter:blur(10px);
box-shadow:0 8px 25px rgba(0,0,0,.4);
transition:.4s;
margin-bottom:15px;
}

.movie-card:hover{
transform:scale(1.03);
background:rgba(255,255,255,0.18);
}

.stButton>button{
background:linear-gradient(90deg,#ff512f,#dd2476);
color:white;
font-size:18px;
font-weight:bold;
border:none;
border-radius:12px;
padding:12px 30px;
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------

st.markdown(
    "<div class='big-title'>🎬 Movie Recommender System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Find movies similar to your favourite movie using NLP</div>",
    unsafe_allow_html=True
)

# ---------------- Dataset ----------------

df = pd.read_csv("cleaned_data.csv")

# ---------------- Similarity ----------------

if os.path.exists("similarity.pkl"):

    similarities = pickle.load(open("similarity.pkl", "rb"))

else:

    st.info("Generating similarity matrix for the first time...")

    with st.spinner("Training AI..."):

        cv = CountVectorizer(
            max_features=10000,
            stop_words="english"
        )

        vector = cv.fit_transform(df["tags"]).toarray()

        similarities = cosine_similarity(vector)

        pickle.dump(similarities, open("similarity.pkl", "wb"))

    st.success("Similarity Matrix Generated Successfully!")

# ---------------- Movie List ----------------

movies = sorted(df["title"].tolist())

st.markdown("## 🎥 Choose a Movie")

name = st.selectbox("", movies)

# ---------------- Functions ----------------

def get_name_by_index(i):

    if 0 <= i < len(df):
        return df.loc[i, "title"]

    return ""


def get_index_from_name(name):

    clean_name = (
        name.strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
    )

    match = df[
        df["title"]
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("-", "", regex=False)
        == clean_name
    ]

    if not match.empty:
        return match.index[0]

    return -1

# ---------------- Recommend ----------------

if st.button("🚀 Recommend Movies"):

    index = get_index_from_name(name)

    if index == -1:

        st.error("Movie Not Found!")

    else:

        progress = st.progress(0)

        for i in range(100):
            progress.progress(i + 1)

        progress.empty()

        st.success(f"Top 5 Recommendations for **{name}**")

        similarity_indexes = list(enumerate(similarities[index]))

        similarity_indexes = sorted(
            similarity_indexes,
            key=lambda x: x[1],
            reverse=True
        )

        col1, col2 = st.columns(2)

        count = 1

        for i in range(1, 6):

            movie = get_name_by_index(similarity_indexes[i][0])

            html = f"""
            <div class='movie-card'>
                <h3>🎬 {movie}</h3>
            </div>
            """

            if count % 2 == 1:

                with col1:
                    st.markdown(html, unsafe_allow_html=True)

            else:

                with col2:
                    st.markdown(html, unsafe_allow_html=True)

            count += 1

st.markdown(
    """
    <br><br>
    <center style="color:white;font-size:16px;">
    ❤️ with Natural Language Processing
    </center>
    """,
    unsafe_allow_html=True
)
