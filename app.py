import os
import streamlit as st
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from ibm_watsonx_ai.foundation_models import Model

# -----------------------
# PDF Reading
# -----------------------
def extract_text_from_pdf_bytes(file_bytes):
    text = ""
    with pdfplumber.open(file_bytes) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

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
# Embeddings
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
st.set_page_config(page_title="EcoLifestyle Agent — PDF Q&A", page_icon="🌱")
st.title("🌱 EcoLifestyle Agent — PDF Q&A (IBM watsonx.ai)")

st.markdown("Upload your PDF and ask questions. The app retrieves relevant passages and uses IBM watsonx.ai to generate answers.")

uploaded_file = st.file_uploader("Upload Project PDF", type=["pdf"])

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "index" not in st.session_state:
    st.session_state.index = None
if "model" not in st.session_state:
    st.session_state.model = None

if uploaded_file is not None:
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf_bytes(uploaded_file)

    if not text.strip():
        st.warning("No extractable text found in PDF.")
    else:
        if not st.session_state.chunks:
            st.session_state.chunks = chunk_text(text)
            st.success(f"Extracted {len(st.session_state.chunks)} chunks.")

        if st.session_state.model is None:
            with st.spinner("Loading embedding model..."):
                st.session_state.model = load_embed_model()

        if st.session_state.embeddings is None:
            with st.spinner("Computing embeddings..."):
                st.session_state.embeddings = embed_texts(st.session_state.model, st.session_state.chunks)

        if st.session_state.index is None:
            with st.spinner("Building index..."):
                st.session_state.index = build_index(st.session_state.embeddings)

# 🔐 Get IBM watsonx.ai credentials from Streamlit Secrets
api_key_input = os.getenv("API_KEY", st.secrets.get("API_KEY", ""))
project_id_input = os.getenv("PROJECT_ID", st.secrets.get("PROJECT_ID", ""))
url_input = os.getenv("URL", st.secrets.get("URL", ""))

query = st.text_input("Ask a question about the PDF")
top_k = st.slider("Top K passages", min_value=1, max_value=8, value=4)

if st.button("Search & Answer") and query.strip():
    if uploaded_file is None:
        st.info("Please upload the PDF first.")
    elif not api_key_input or not url_input or not project_id_input:
        st.warning("IBM watsonx.ai credentials missing. Set them in Streamlit Secrets.")
    else:
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
            answer = synthesize_with_watsonx(api_key_input, url_input, project_id_input, query, top_chunks)
        st.subheader("Agent Answer")
        st.write(answer)


