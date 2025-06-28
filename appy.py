import pickle
import streamlit as st
import requests
import pandas as pd
import os
from huggingface_hub import hf_hub_download

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# Load movies_dict as before
movies_dict = pickle.load(open('movies_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

# Download similarity.pkl from Hugging Face if not present locally
@st.cache_resource(show_spinner="Downloading similarity matrix from Hugging Face...")
def load_similarity():
    # This will download and cache the file in ~/.cache/huggingface/hub/
    file_path = hf_hub_download(
        repo_id="dhr-uuu34/movie-recommender",
        filename="similarity.pkl"
    )
    with open(file_path, 'rb') as f:
        similarity = pickle.load(f)
    return similarity

similarity = load_similarity()

st.title("MOVIE RECOMMENDATIONS")

selected_movie_name = st.selectbox('Select a movie:', movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)
