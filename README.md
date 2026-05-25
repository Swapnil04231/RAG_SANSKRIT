Sanskrit Document Retrieval-Augmented Generation (RAG) System
Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system for Sanskrit documents using CPU-only inference.

The system:

Loads Sanskrit documents
Creates embeddings
Stores vectors in FAISS
Retrieves relevant chunks
Generates responses using TinyLlama via Ollama

The project supports:

Sanskrit queries
Transliteration queries
English queries
Features
Sanskrit document ingestion
PDF and TXT support
FAISS vector database
Multilingual embeddings
TinyLlama integration
CPU-only inference
Gradio web interface
Project Structure
RAG_Sanskrit_Swapnil/
│
├── code/
│   ├── ingest.py
│   ├── rag_pipeline.py
│   └── app.py
│
├── data/
│   └── sanskrit.txt
│
├── faiss_index/
│
├── report/
│   └── report.pdf
│
├── screenshots/
│
├── requirements.txt
│
└── README.md


Installation
1. Clone Repository
git clone <your_github_repo>
2. Create Virtual Environment
python -m venv venv
3. Activate Environment
Windows
venv\Scripts\activate
4. Install Dependencies
pip install -r requirements.txt
Install Ollama

Download and install:

Ollama Official Website

Pull TinyLlama model:

ollama pull tinyllama
Running the Project
Step 1 — Create FAISS Index
cd code
python ingest.py
Step 2 — Start TinyLlama
ollama run tinyllama
Step 3 — Run Web Application

Open another terminal:

cd code
python app.py
Example Queries
Sanskrit
धर्मः किम्?
योगस्य महत्वं किम्?
Transliteration
dharmah kim?
English
What is yoga?
System Architecture
Documents
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS Vector Store
   ↓
Retriever
   ↓
TinyLlama (Ollama)
   ↓
Generated Response
Technologies Used
Technology	Purpose
Python	Programming
LangChain	RAG framework
FAISS	Vector database
TinyLlama	LLM
Ollama	Local inference
Gradio	UI
PyMuPDF	PDF extraction
CPU Optimization
TinyLlama 1.1B model
Small chunk size
Top-K retrieval
FAISS vector search
Quantized Ollama models
Future Improvements
Sanskrit-specific embeddings
Hybrid retrieval
Fine-tuned Sanskrit LLM
Better OCR preprocessing
Author

Swapnil Bagale