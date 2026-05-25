import os
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# LOAD PDF FILE
# =========================
def load_pdf(path):
    text = ""

    doc = fitz.open(path)

    for page in doc:
        text += page.get_text()

    return text


# =========================
# LOAD TXT FILE
# =========================
def load_txt(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# =========================
# LOAD ALL DOCUMENTS
# =========================
def load_documents(data_folder):
    all_text = ""

    for filename in os.listdir(data_folder):

        file_path = os.path.join(data_folder, filename)

        # PDF
        if filename.endswith(".pdf"):
            print(f"Loading PDF: {filename}")
            all_text += load_pdf(file_path)

        # TXT
        elif filename.endswith(".txt"):
            print(f"Loading TXT: {filename}")
            all_text += load_txt(file_path)

    return all_text


# =========================
# MAIN PIPELINE
# =========================
def main():

    # Step 1: Load documents
    data_folder = "../data"

    print("Loading Sanskrit documents...")

    raw_text = load_documents(data_folder)

    print("Documents loaded successfully!")

    # Step 2: Split text into chunks
    print("Splitting text into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.create_documents([raw_text])

    print(f"Total chunks created: {len(chunks)}")

    # Step 3: Create embeddings
    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Step 4: Create FAISS vector DB
    print("Creating FAISS vector database...")

    db = FAISS.from_documents(chunks, embeddings)

    # Step 5: Save vector DB
    db.save_local("../faiss_index")

    print("FAISS index saved successfully!")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()