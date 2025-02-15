import time
from pathlib import Path
from typing import Tuple, Dict, List

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

from core.retrieval import EnhancedRetriever
from models.llm import generate_response
from config import app_config
from utils.logger import logger
from document_handler import DocumentHandler

print("HF_API_KEY loaded:", "✅ Valid" if app_config.hf_api_key.startswith("hf_") else "❌ Invalid")

retriever = EnhancedRetriever()
doc_handler = DocumentHandler()

def create_plots(metrics_history: List[Dict]) -> Tuple[plt.Figure, plt.Figure]:
    time_fig, ax1 = plt.subplots(figsize=(8, 4))
    queries = list(range(1, len(metrics_history) + 1))

    ax1.plot(queries, [m['total_time'] for m in metrics_history], label='Total Time', marker='o')
    ax1.plot(queries, [m.get('retrieval_time', 0) for m in metrics_history], label='Retrieval Time', marker='x')
    ax1.set_xlabel('Query Sequence')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Processing Time Timeline')
    ax1.legend()
    ax1.grid(True)

    accuracy_fig, ax2 = plt.subplots(figsize=(8, 4))
    width = 0.35
    x = range(len(metrics_history))

    ax2.bar(x, [m.get('precision@k', 0) for m in metrics_history], width, label='Precision@k')
    ax2.bar([p + width for p in x], [m.get('recall@k', 0) for m in metrics_history], width, label='Recall@k')
    ax2.set_xlabel('Query Sequence')
    ax2.set_ylabel('Score')
    ax2.set_title('Retrieval Performance')
    ax2.set_xticks([p + width/2 for p in x])
    ax2.set_xticklabels(queries)
    ax2.legend()
    ax2.grid(True)

    return time_fig, accuracy_fig

def create_chat_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="RAG Performance meter") as demo:
        gr.Markdown("# 🚀 RAG Performance Meter")

        metrics_history = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                model_choice = gr.Dropdown(
                    choices=[
                        ("DeepSeek-R1", "  "), #add your model url endpoint 
                        ("OpenAi o1", ""),#add your model url endpoint 
                        ("Gemini 2.0", ""), #add your model url endpoint 
                        ("Claude 3.5", "") #add your model url endpoint 
                    ],
                    value="", #add your model url endpoint 
                    label="Model Selection"
                )

                rag_toggle = gr.Checkbox(
                    value=True,
                    label="Enable RAG (Requires document upload)"
                )
                file_upload = gr.File(
                    label=" 📁Upload Document (Optional for RAG)",
                    type="filepath",
                    interactive=True
                )

                with gr.Accordion("Advanced Settings", open=False):
                    chunk_size = gr.Slider(256, 2048, value=1024, label="Chunk Size")
                    chunk_overlap = gr.Slider(0, 512, value=128, label="Chunk Overlap")
                    top_k = gr.Slider(1, 10, value=3, label="Top K Results")
                
                gr.Markdown("## ⏱ Processing Timeline")
                time_plot = gr.Plot()

                gr.Markdown("## 🎯 Accuracy Metrics")
                accuracy_plot = gr.Plot()
                
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=500,
                    value=[],
                    type="messages"
                )
                prompt = gr.Textbox(
                    label="Your Question",
                    placeholder="Type your question here...",
                    lines=3
                )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.ClearButton([prompt, chatbot, file_upload])

                gr.Markdown("## 📑 Referensi Dokumen")
                chunk_references_table = gr.DataFrame(
                    headers=["Chunk ID", "Skor", "Konten"],
                    datatype=["number", "number", "str"],
                    interactive=False
                )

            with gr.Column(scale=1):
                gr.Markdown("## 📊 Performance Dashboard")
                metrics_df = gr.DataFrame(
                    headers=["Metric", "Value"],
                    datatype=["str", "number"],
                    interactive=False
                )

                gr.Markdown("### 🔢 Token Usage")
                token_stats = gr.DataFrame(
                    headers=["Type", "Count"],
                    datatype=["str", "number"]
                )

        def process_query(
            prompt: str,
            history: List[Dict],
            model: str,
            use_rag: bool,
            file: str,
            top_k_val: int,
            metrics_hist: List[Dict]
        ) -> Tuple[List[Dict], pd.DataFrame, List[Dict], plt.Figure, plt.Figure, pd.DataFrame, pd.DataFrame]:

            start_time = time.time()
            context = ""
            retrieved = []
            metrics = {}
            response = ""
            tokens = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }

            try:
                if use_rag and file:
                    file_hash = doc_handler.get_file_hash(file)
                    chunks = doc_handler.load_document(file)

                    cleaned_chunks = [
                        {"page_content": doc_handler.sanitize_content(chunk.page_content)}
                        for chunk in chunks
                    ]

                    retriever.process_documents(cleaned_chunks, file_hash)
                    retrieved, metrics = retriever.retrieve(prompt, top_k_val)

                    context_parts = []
                    for i, chunk in enumerate(retrieved):
                        safe_content = doc_handler.sanitize_content(chunk.page_content)
                        context_parts.append(f"Context {i+1}: {safe_content}")
                    context = "\n".join(context_parts)

                time.sleep(1.5 if app_config.hf_api_key.startswith("hf_") else 3)

                response, token_usage = generate_response(prompt, model, context)
                response = str(response).replace("```", "```\n").strip()
                tokens = {
                    "input_tokens": token_usage.get('prompt_tokens', 0),
                    "output_tokens": token_usage.get('completion_tokens', 0),
                    "total_tokens": token_usage.get('total_tokens', 0)
                }

            except UnicodeDecodeError as ude:
                error_msg = f"🚨 File encoding error: {str(ude)}"
                logger.error(error_msg)
                response = "Error: Invalid file encoding. Please use UTF-8 encoded files."

            except ValueError as ve:
                error_msg = f"🚨 Document error: {str(ve)}"
                logger.error(error_msg)
                response = f"Document processing error: {str(ve)}"

            except Exception as e:
                error_msg = f"🚨 System error: {str(e)}"
                logger.error(error_msg)
                response = error_msg[:500]

                if "429" in str(e):
                    response += "\n\n🔧 Solution: Please wait 30 seconds before next request"
                elif "401" in str(e):
                    response += "\n\n🔧 Solution: Check your HF_API_KEY in config.py"

            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": response})

            total_time = round(time.time() - start_time, 2)
            metrics_data = [
                ("Total Time (s)", total_time),
                ("Retrieval Time (s)", round(metrics.get('retrieval_time', 0), 2)),
                ("Chunks Used", len(retrieved)),
                ("Precision@k", round(metrics.get('precision@k', 0), 2)),
                ("Recall@k", round(metrics.get('recall@k', 0), 2)),
                ("F1 Score", round(metrics.get('f1_score', 0), 2)),
                ("RAG Mode", "Active" if use_rag else "Inactive"),
                ("Document Uploaded", "Yes" if file else "No")
            ]

            token_df = pd.DataFrame({
                "Type": ["Input Tokens", "Output Tokens", "Total Tokens"],
                "Count": [tokens['input_tokens'], tokens['output_tokens'], tokens['total_tokens']]
            })

            new_metrics = {
                "total_time": total_time,
                "retrieval_time": metrics.get('retrieval_time', 0),
                "precision@k": metrics.get('precision@k', 0),
                "recall@k": metrics.get('recall@k', 0),
                "f1_score": metrics.get('f1_score', 0),
                "tokens": tokens
            }
            metrics_hist.append(new_metrics)

            time_fig, accuracy_fig = create_plots(metrics_hist)

            if use_rag and file and retrieved:
                chunk_refs_df = pd.DataFrame([{
                    "Chunk ID": idx + 1,
                    "Skor": chunk.metadata.get("score", 0),
                    "Konten": doc_handler.sanitize_content(chunk.page_content)
                } for idx, chunk in enumerate(retrieved)])
            else:
                chunk_refs_df = pd.DataFrame(columns=["Chunk ID", "Skor", "Konten"])

            return (
                history,
                pd.DataFrame(metrics_data, columns=["Metric", "Value"]),
                metrics_hist,
                time_fig,
                accuracy_fig,
                token_df,
                chunk_refs_df
            )

        submit_btn.click(
            fn=process_query,
            inputs=[prompt, chatbot, model_choice, rag_toggle, file_upload, top_k, metrics_history],
            outputs=[chatbot, metrics_df, metrics_history, time_plot, accuracy_plot, token_stats, chunk_references_table]
        )

    return demo

if __name__ == "__main__":
    app = create_chat_interface()
    app.queue(
        default_concurrency_limit=1,
        max_size=5
    )
    app.launch(
        server_port=7860,
        show_error=True
    )
