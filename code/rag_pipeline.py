from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import ollama

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Loading FAISS index...")

db = FAISS.load_local(
    "../faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully!")

def ask_question(query):

    docs = db.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer based on the Sanskrit context below.

Context:
{context}

Question:
{query}

Answer:
"""
    response = ollama.chat(
        model="tinyllama",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response["message"]["content"]

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