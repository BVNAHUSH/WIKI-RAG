import streamlit as st
import wikipedia
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------------- CACHE MODELS ----------------------
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"



@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def load_qa_pipeline():
    qa_model_name = "deepset/roberta-base-squad2"
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    return pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

# ---------------------- FUNCTIONS ----------------------
def get_wikipedia_content(topic):
    try:
        page = wikipedia.page(topic)
        return page.content
    except wikipedia.exceptions.PageError:
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        st.warning(f"Ambiguous topic. Try: {e.options}")
        return None


def split_text(text, chunk_size=256, chunk_overlap=20):
    tokens = tokenizer.tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.convert_tokens_to_string(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - chunk_overlap
    return chunks

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="RAG Pipeline Demo", page_icon="🔍", layout="wide")
st.title("🔍 Retrieval-Augmented Generation (RAG) Demo")
st.write("**Ask questions about any topic from Wikipedia!**")

# Load models (cached)
tokenizer = load_tokenizer()
embedding_model = load_embedding_model()
qa_pipeline = load_qa_pipeline()

# User inputs
topic = st.text_input("Enter a topic:", placeholder="e.g., Artificial Intelligence")
question = st.text_input("Ask a question about the topic:", placeholder="e.g., Who is known as the father of AI?")
k = st.slider("Number of chunks to retrieve:", 1, 5, 3)

# Main processing
if st.button("Get Answer"):
    if not topic:
        st.error("Please enter a topic.")
    else:
        with st.spinner("Fetching Wikipedia content..."):
            document = get_wikipedia_content(topic)
            if not document:
                st.error("Could not retrieve information.")
                st.stop()

        with st.spinner("Processing chunks and embeddings..."):
            chunks = split_text(document)
            st.write(f"**Number of chunks created:** {len(chunks)}")

            embeddings = embedding_model.encode(chunks)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))

            query_embedding = embedding_model.encode([question])
            distances, indices = index.search(np.array(query_embedding), k)
            retrieved_chunks = [chunks[i] for i in indices[0]]

        st.subheader("Retrieved Chunks")
        for i, chunk in enumerate(retrieved_chunks, start=1):
            st.markdown(f"**Chunk {i}:** {chunk}")

        with st.spinner("Generating answer..."):
            context = " ".join(retrieved_chunks)
            answer = qa_pipeline(question=question, context=context)
            st.success(f"**Answer:** {answer['answer']}")

        # Option to download retrieved chunks
        st.download_button(
            label="📥 Download Retrieved Chunks",
            data="\n\n".join(retrieved_chunks),
            file_name=f"{topic}_chunks.txt",
            mime="text/plain"
        )
