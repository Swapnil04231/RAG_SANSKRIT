from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import ollama


# =========================
# LOAD EMBEDDING MODEL
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# =========================
# LOAD FAISS DATABASE
# =========================
print("Loading FAISS index...")

db = FAISS.load_local(
    "../faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully!")


# =========================
# ASK QUESTION FUNCTION
# =========================
def ask_question(query):

    # Step 1: Retrieve relevant chunks
    docs = db.similarity_search(query, k=3)

    # Step 2: Combine retrieved context
    context = "\n".join([doc.page_content for doc in docs])

    # Step 3: Create prompt
    prompt = f"""
Answer based on the Sanskrit context below.

Context:
{context}

Question:
{query}

Answer:
"""

    # Step 4: Send to TinyLlama
    response = ollama.chat(
        model="tinyllama",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Step 5: Return response
    return response["message"]["content"]


# =========================
# TERMINAL CHAT LOOP
# =========================
if __name__ == "__main__":

    print("\nSanskrit RAG System Ready!")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("Ask Question: ")

        if query.lower() == "exit":
            break

        answer = ask_question(query)

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "=" * 50 + "\n")