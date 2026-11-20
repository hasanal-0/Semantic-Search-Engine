# Week 13 Interactive: Semantic Search Engine 
## 🎬 Movie Plot Search

By Hasan Al-Quaid

Description:
\
A semantic search engine for movie plots. Describe what happens in a movie in your own words, like *"a robbery that goes terribly wrong"* or *"a man stranded alone on another planet"*, and it finds the movies whose plots match the **meaning**, even when they share no exact words with your search.

**Live app:** *https://semantic-search-engine-hasan.streamlit.app/*

## How it works

1. **Dataset:** Wikipedia movie plots ([`Coder-Dragon/wikipedia-movies`](https://huggingface.co/datasets/Coder-Dragon/wikipedia-movies) on Hugging Face). I used a fixed random sample of 1,500 American movies from 1990 on (seed 20), which keeps the database small enough for GitHub.
2. **Chunk:** Each plot is split into chunks of 150 words, with 30 words of overlap between neighboring chunks. The embedding model only reads about 256 tokens (~180 words), so a long plot embedded whole would be cut off. The overlap keeps sentences that fall on a chunk boundary from being split apart.
3. **Embed:** Each chunk becomes a 384-number vector using `all-MiniLM-L6-v2`, the same model as the starter notebook, run through Chroma's built-in embedding function.
4. **Store:** The chunks, their vectors, and metadata (title, year, genre) are saved in a **Chroma** vector database in `chromadb/`, using cosine similarity.
5. **Search:** The app embeds your query with the same model and asks Chroma for the closest chunks. It keeps only the best-matching chunk per movie and shows the top results with a similarity score (1 = identical meaning).

## Files

| File | What it does |
|---|---|
| `build_db.py` | Downloads the data, chunks it, embeds it, and saves the Chroma database. Run once. |
| `chromadb/` | The saved vector database the app loads |
| `app.py` | The Streamlit search app |
| `requirements.txt` | Packages the app needs (pinned) |

## Run it yourself

```bash
# 1. Build the database (one time, takes a few minutes)
pip install chromadb==1.5.9 datasets
python build_db.py

# 2. Run the app
pip install -r requirements.txt
streamlit run app.py
```

Use the same `chromadb` version for building and running. A database saved by one version may not load in another.

## Limitations

- **Only 1,500 movies** (American, 1990+), so a movie you're thinking of may not be in it.
- **Plots come from Wikipedia** and contain spoilers.
- **Small embedding model.** `all-MiniLM-L6-v2` is fast and light but less precise than larger models. Very specific or unusual queries may return loose matches.
