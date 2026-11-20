"""
Movie Plot Search: a semantic search engine over Wikipedia movie plots.

Run locally:  streamlit run app.py
"""
import chromadb
import streamlit as st
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


@st.cache_resource
def load_search_tools():
    """
    Load the saved Chroma collection and the embedding model once, then reuse them

    Returns:
        tuple: The Chroma collection and the embedding function
    """
    client = chromadb.PersistentClient(path="chromadb")
    return client.get_collection("movies"), DefaultEmbeddingFunction()


def semantic_search(query, num_movies):
    """
    Find the movies whose plots are most similar in meaning to the query

    Parameters:
        query (str): What the user typed, e.g. "a heist that goes wrong"
        num_movies (int): How many different movies to return

    Returns:
        list: One dict per movie with title, year, genre, score, and the matching passage
    """
    collection, embed = load_search_tools()
    results = collection.query(
        query_embeddings=embed([query]),
        n_results=num_movies * 5,   # several chunks can come from the same movie
    )

    movies, seen_titles = [], set()
    for passage, meta, distance in zip(results["documents"][0],
                                       results["metadatas"][0],
                                       results["distances"][0]):
        if meta["title"] in seen_titles:
            continue                # keep only the best chunk per movie
        seen_titles.add(meta["title"])
        movies.append({**meta, "passage": passage, "score": 1 - distance})
        if len(movies) == num_movies:
            break
    return movies


st.set_page_config(page_title="Movie Plot Search", page_icon="🎬")
st.title("🎬 Movie Plot Search")
st.write("Describe a plot in your own words. The search matches **meaning**, not exact keywords, "
         "across 1,500 American movies (1990+) from Wikipedia.")

query = st.text_input("What happens in the movie?",
                      placeholder="e.g. a robbery that goes terribly wrong")
num_movies = st.slider("Number of results", 1, 10, 5)

MIN_SCORE = 0.30   # hide results less similar than this
if query:
    results = [m for m in semantic_search(query, num_movies) if m["score"] >= MIN_SCORE]
    if not results:
        st.warning("No good matches. Try describing a plot, e.g. 'a heist that goes wrong'.")
    for movie in results:
        st.subheader(f"{movie['title']} ({movie['year']})")
        st.markdown(f"**Similarity: {movie['score']:.0%}**  ·  Genre: {movie['genre']}")
        st.progress(min(max(movie["score"], 0.0), 1.0))
        st.write(movie["passage"])
        st.divider()
