import gradio as gr

from rag_pipeline import ask_question

iface = gr.Interface(
    fn=ask_question,

    inputs=gr.Textbox(
        lines=2,
        placeholder="Enter Sanskrit question here..."
    ),

    outputs=gr.Textbox(
        lines=10
    ),

    title="Sanskrit RAG System",
    description="Ask questions from Sanskrit documents using Retrieval-Augmented Generation (RAG)."
)

iface.launch()