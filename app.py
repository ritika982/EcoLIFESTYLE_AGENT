import os
import streamlit as st
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from ibm_watsonx_ai.foundation_models import Model

# -----------------------
# IBM watsonx.ai credentials
# -----------------------
API_KEY = "UZ2TVy83LurEvlPjJlXWXuYXU8L1ba5lTUafb2T4KKxG"  # replace with your API key
PROJECT_ID = "cc79370a-2d49-4f0f-afc7-13f0c8749038"  # replace with your project id
URL = "https://us-south.ml.cloud.ibm.com"  # replace with your service url

# -----------------------
# Load PDF automatically from repo
# -----------------------
PDF_PATH = os.path.join(os.path.dirname(__file__), "eco_tips.pdf")

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

pdf_text = extract_text_from_pdf(PDF_PATH)

def chunk_text(text, chunk_size=700, overlap=150):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

# -----------------------
# Embeddings & Search
# -----------------------
@st.cache_resource(show_spinner=False)
def load_embed_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def embed_texts(model, texts):
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def build_index(embeddings):
    nn = NearestNeighbors(n_neighbors=8, metric="cosine")
    nn.fit(embeddings)
    return nn

def search_top_k(index, embeddings, query_emb, k=5):
    distances, indices = index.kneighbors([query_emb], n_neighbors=k)
    return indices[0].tolist(), distances[0].tolist()

# -----------------------
# IBM watsonx.ai synthesis
# -----------------------
def synthesize_with_watsonx(api_key, url, project_id, query, top_chunks):
    model_id = "ibm/granite-13b-chat-v2"
    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 200,
        "temperature": 0.1
    }
    model = Model(
        model_id=model_id,
        params=parameters,
        credentials={"apikey": api_key, "url": url},
        project_id=project_id
    )

    context_texts = []
    for idx, ch in top_chunks:
        context_texts.append(f"[Chunk {idx}]: {ch[:1000]}")
    prompt = (
        "You are an AI assistant answering questions based only on the provided document chunks.\n"
        "If the answer is not present, say so. Be concise.\n\n"
        f"Question: {query}\n\n"
        "Document chunks:\n" + "\n\n".join(context_texts) + "\n\n"
        "Answer:"
    )

    try:
        response = model.generate_text(prompt=prompt)
        return response
    except Exception as e:
        return f"Watsonx.ai API call failed: {e}"

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="EcoLifestyle Agent — Preloaded PDF", page_icon="🌱")
st.title("🌱 EcoLifestyle Agent — Preloaded PDF")

st.markdown("Ask questions about the eco lifestyle PDF stored in the app.")

# Prepare chunks and embeddings only once
if "chunks" not in st.session_state:
    st.session_state.chunks = chunk_text(pdf_text)
    st.success(f"Loaded {len(st.session_state.chunks)} chunks from PDF.")

if "model" not in st.session_state:
    with st.spinner("Loading embedding model..."):
        st.session_state.model = load_embed_model()

if "embeddings" not in st.session_state:
    with st.spinner("Computing embeddings..."):
        st.session_state.embeddings = embed_texts(st.session_state.model, st.session_state.chunks)

if "index" not in st.session_state:
    with st.spinner("Building search index..."):
        st.session_state.index = build_index(st.session_state.embeddings)

# Get question
query = st.text_input("Ask a question about the PDF")
top_k = st.slider("Top K passages", min_value=1, max_value=8, value=4)

if st.button("Search & Answer") and query.strip():
    query_emb = embed_texts(st.session_state.model, [query])[0]
    indices, distances = search_top_k(st.session_state.index, st.session_state.embeddings, query_emb, k=top_k)

    st.subheader("Retrieved Passages")
    top_chunks = []
    for rank, idx in enumerate(indices, start=1):
        idx = int(idx)
        chunk_text_display = st.session_state.chunks[idx]
        top_chunks.append((idx, chunk_text_display))
        st.markdown(f"**{rank}. Chunk {idx}** (score: {float(distances[rank-1]):.4f})")
        st.write(chunk_text_display)
        st.write("---")

    with st.spinner("Generating answer with IBM watsonx.ai..."):
        answer = synthesize_with_watsonx(API_KEY, URL, PROJECT_ID, query, top_chunks)
    st.subheader("Agent Answer")
    st.write(answer)




