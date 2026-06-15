import os
from typing import List, Dict
from anthropic import Anthropic
from src.vectorstore import VectorStore

client = Anthropic()

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on provided document context. "
    "Only use the information from the context to answer. "
    "If the answer is not in the context, say so clearly."
)


def generate_answer(
    query: str,
    vectorstore: VectorStore,
    chat_history: List[Dict[str, str]],
) -> str:
    context_chunks = vectorstore.query(query)
    context = "\n\n".join(context_chunks)

    messages: List[Dict[str, str]] = []
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Context from document:\n{context}\n\nQuestion: {query}",
    })

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return response.content[0].text
