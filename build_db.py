"""
Build the Chroma vector database for the movie plot search app.

Run once on your own computer:  python build_db.py
It downloads the movie plots, chunks them, embeds each chunk, and saves
everything to the chromadb/ folder. Commit that folder to GitHub so the
Streamlit app can load it without rebuilding.
"""
import shutil

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from datasets import load_dataset

DB_PATH = "chromadb"
COLLECTION_NAME = "movies"
NUM_MOVIES = 1500      # keeps the database small enough for GitHub
CHUNK_WORDS = 150      # the embedding model reads ~256 tokens (~180 words) max
OVERLAP_WORDS = 30     # overlap so a sentence cut at a chunk edge appears in both chunks
SEED = 20


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS):
    """
    Split a long text into overlapping chunks of words

    Parameters:
        text (str): The text to split
        chunk_words (int): Number of words in each chunk
        overlap_words (int): Number of words shared by neighboring chunks

    Returns:
        list: A list of text chunks
    """
    words = text.split()
    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks


# 1. Load the dataset: Wikipedia movie plots (Hugging Face: Coder-Dragon/wikipedia-movies)
df = load_dataset("Coder-Dragon/wikipedia-movies", split="train").to_pandas()
print(f"Loaded {len(df)} movies. Columns: {list(df.columns)}")

# 2. Keep American movies from 1990 on, then take a fixed random sample
df = df[(df["Origin/Ethnicity"] == "American") & (df["Release Year"] >= 1990)]
df = df.dropna(subset=["Title", "Plot"])
df = df.sample(n=min(NUM_MOVIES, len(df)), random_state=SEED).reset_index(drop=True)
print(f"Using {len(df)} American movies from 1990 on")

# 3. Chunk every plot, keeping track of which movie each chunk came from
chunks, metadatas, ids = [], [], []
for i, row in df.iterrows():
    for j, chunk in enumerate(chunk_text(row["Plot"])):
        chunks.append(chunk)
        metadatas.append({
            "title": row["Title"],
            "year": int(row["Release Year"]),
            "genre": str(row["Genre"]),
        })
        ids.append(f"{i}-{j}")
print(f"Split plots into {len(chunks)} chunks")

# 4. Embed the chunks with all-MiniLM-L6-v2 (same model as the starter notebook)
embed = DefaultEmbeddingFunction()

# 5. Store chunks, embeddings, and metadata in a Chroma database saved to disk
shutil.rmtree(DB_PATH, ignore_errors=True)          # start fresh every time
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

BATCH = 500
for start in range(0, len(chunks), BATCH):
    batch = chunks[start:start + BATCH]
    collection.add(
        documents=batch,
        embeddings=embed(batch),
        metadatas=metadatas[start:start + BATCH],
        ids=ids[start:start + BATCH],
    )
    print(f"  embedded {min(start + BATCH, len(chunks))}/{len(chunks)} chunks")

print(f"Done. Saved {collection.count()} chunks to {DB_PATH}/")
