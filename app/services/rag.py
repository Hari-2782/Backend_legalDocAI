from typing import List, Dict, Any
from app.services.embedding import query_vectors


def retrieve_snippets(question: str, file_id: str, top_k: int = 5):
    res = query_vectors(question, file_id=file_id, top_k=top_k)
    docs, snippets = [], []
    for doc_id, doc_text, meta, score in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        docs.append({"chunk_id": doc_id, "text": doc_text, "meta": meta, "score": score})
        snippets.append(doc_text)
    return docs, snippets


def build_prompt(question: str, snippets: List[str]) -> str:
    context = "\n\n".join([f"Snippet {i+1}: {s}" for i, s in enumerate(snippets)])
    prompt = (
        "You are a legal assistant. Answer the question using ONLY the snippets below. "
        "If the document does not contain the answer, respond 'Not stated in document'.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer in plain English, cite snippet numbers in [S#], list any risks, and give confidence (High/Medium/Low)."
    )
    return prompt