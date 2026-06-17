import os
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq
from src.vectorstore import VectorStore

load_dotenv()

# load_dotenv() must run before the client is created, otherwise GROQ_API_KEY
# is still unset and Groq() raises on import.
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
    """Answer a question using RAG: retrieve relevant chunks, then ask the LLM."""
    context_chunks = vectorstore.query(query)
    context = "\n\n".join(context_chunks)

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Only keep the last 6 messages so the prompt stays short and cheap,
    # while still giving the model some conversational memory.
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Context from document:\n{context}\n\nQuestion: {query}",
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=messages,
    )

    return response.choices[0].message.content
