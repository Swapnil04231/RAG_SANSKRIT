import os
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_pdf(path):
    text = ""

    doc = fitz.open(path)

    for page in doc:
        text += page.get_text()
    return text

def load_txt(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def load_documents(data_folder):
    all_text = ""

    for filename in os.listdir(data_folder):

        file_path = os.path.join(data_folder, filename)

        if filename.endswith(".pdf"):
            print(f"Loading PDF: {filename}")
            all_text += load_pdf(file_path)

        elif filename.endswith(".txt"):
            print(f"Loading TXT: {filename}")
            all_text += load_txt(file_path)

    return all_text

def main():

    data_folder = "../data"

    print("Loading Sanskrit documents...")

    raw_text = load_documents(data_folder)

    print("Documents loaded successfully!")
    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.create_documents([raw_text])

    print(f"Total chunks created: {len(chunks)}")
    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("Creating FAISS vector database...")

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("../faiss_index")

    print("FAISS index saved successfully!")

if __name__ == "__main__":
    main()