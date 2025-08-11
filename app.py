# app.py
import os
import streamlit as st
import fitz                 # pymupdf
import numpy as np
from sentence_transformers import SentenceTransformer
import tempfile
import time

# Optional: faiss for fast vector search
try:
    import faiss
    USE_FAISS = True
except Exception:
    from sklearn.neighbors import NearestNeighbors
    USE_FAISS = False

# OpenAI client
try:
    import openai
except Exception:
    openai = None

# -----------------------
# Helpers: PDF, chunking
# -----------------------
def extract_text_from_pdf_bytes(file_bytes):
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def chunk_text(text, chunk_size=700, overlap=150):
    """Chunk by characters with overlap (simple, but effective)."""
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
# Embedding model & index
# -----------------------
@st.cache_resource(show_spinner=False)
def load_embed_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def embed_texts(model, texts):
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def build_index(embeddings):
    d = embeddings.shape[1]
    if USE_FAISS:
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        return ("faiss", index)
    else:
        nn = NearestNeighbors(n_neighbors=8, metric="cosine")
        nn.fit(embeddings)
        return ("sklearn", nn)

def search_top_k(index_obj, embeddings, query_emb, k=5):
    backend, index = index_obj
    if backend == "faiss":
        D, I = index.search(np.array([query_emb]), k)
        return I[0].tolist(), D[0].tolist()
    else:
        dists, inds = index.kneighbors([query_emb], n_neighbors=k)
        return inds[0].tolist(), dists[0].tolist()

# -----------------------
# Synthesis via OpenAI
# -----------------------
def synthesize_with_openai(api_key, query, top_chunks, model="gpt-4o-mini", max_tokens=256, temperature=0.2):
    """
    Calls OpenAI ChatCompletion (Chat API) to summarize/synthesize the final answer.
    - api_key: your OPENAI_API_KEY
    - top_chunks: a list of (index, chunk_text) tuples ordered by relevance
    """
    if openai is None:
        return "OpenAI package not installed. Install `openai` to enable synthesis."

    openai.api_key = api_key
    # Build a short context with top chunks; keep token limits in mind
    context_texts = []
    for idx, ch in top_chunks:
        # keep each chunk to a reasonable length
        context_texts.append(f"[Chunk {idx}]: {ch[:1000]}")

    system_prompt = (
        "You are an assistant that answers concise, factual questions based only on the provided document chunks. "
        "If the answer is not present in the chunks, say you couldn't find it. "
        "Cite chunk indices when helpful, and answer in 2-5 sentences."
    )

    user_prompt = (
        f"Question: {query}\n\n"
        f"Document chunks:\n" + "\n\n".join(context_texts) + "\n\n"
        "Answer the question concisely and cite chunk numbers if you used them."
    )

    try:
        # Use the chat completion endpoint
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        answer = resp["choices"][0]["message"]["content"].strip()
        return answer
    except Exception as e:
        return f"OpenAI API call failed: {e}"

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="EcoLifestyle Agent — PDF Q&A (with synthesis)", page_icon="🌱")
st.title("🌱 EcoLifestyle Agent — PDF Q&A (Embeddings + OpenAI Synthesis)")

st.markdown(
    "Upload your project PDF, the app will retrieve relevant passages and synthesize a concise answer using a small LLM (OpenAI)."
)

uploaded_file = st.file_uploader("Upload Project PDF (Project_EcoLifestyle.pdf)", type=["pdf"])

# app state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "index" not in st.session_state:
    st.session_state.index = None
if "model" not in st.session_state:
    st.session_state.model = None

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf_bytes(bytes_data)

    if not text.strip():
        st.warning("No extractable text found in PDF. If your PDF is scanned images, OCR is needed (not included here).")
    else:
        if not st.session_state.chunks:
            st.session_state.chunks = chunk_text(text, chunk_size=700, overlap=150)
            st.success(f"Extracted {len(st.session_state.chunks)} chunks from PDF.")
        # load embedding model
        if st.session_state.model is None:
            with st.spinner("Loading embedding model (downloads once)..."):
                st.session_state.model = load_embed_model()
        # compute embeddings
        if st.session_state.embeddings is None:
            with st.spinner("Computing embeddings for chunks... (may take a moment)"):
                st.session_state.embeddings = embed_texts(st.session_state.model, st.session_state.chunks)
        # build index
        if st.session_state.index is None:
            with st.spinner("Building vector index..."):
                st.session_state.index = build_index(st.session_state.embeddings)
            st.success("Vector index is ready. Ask a question below.")

# Query UI
query = st.text_input("Ask a question about the PDF (e.g., 'What tech was used?', 'What is the problem statement?')")

col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("Top K passages to retrieve", min_value=1, max_value=8, value=4)
with col2:
    model_option = st.selectbox("OpenAI model (synthesis)", options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)

api_key_input = st.text_input("OpenAI API Key (paste here or set OPENAI_API_KEY env var)", type="password")
if not api_key_input:
    api_key_input = os.getenv("OPENAI_API_KEY", "")

if st.button("Search & Synthesize") and query.strip():
    if uploaded_file is None:
        st.info("Please upload the PDF first.")
    elif st.session_state.index is None:
        st.info("Index not ready yet. Wait for embeddings/index to finish.")
    else:
        # embed query
        query_emb = embed_texts(st.session_state.model, [query])[0]
        indices, distances = search_top_k(st.session_state.index, st.session_state.embeddings, query_emb, k=top_k)

        st.subheader("Retrieved passages (most relevant first)")
        top_chunks = []
        for rank, idx in enumerate(indices, start=1):
            idx = int(idx)
            chunk_text = st.session_state.chunks[idx]
            top_chunks.append((idx, chunk_text))
            st.markdown(f"**{rank}. Chunk {idx}** (score: {float(distances[rank-1]):.4f})")
            st.write(chunk_text)
            st.write("---")

        # Synthesize using OpenAI if key available
        if api_key_input:
            with st.spinner("Synthesizing answer with OpenAI..."):
                answer = synthesize_with_openai(api_key_input, query, top_chunks, model=model_option)
            st.subheader("Agent — Synthesized Answer")
            st.write(answer)
        else:
            st.warning("No OpenAI API key provided. Raw retrieval shown above. Provide OPENAI_API_KEY to generate a natural answer.")
