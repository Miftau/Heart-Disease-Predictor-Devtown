from groq import Groq
from openai import OpenAI
from config import Config
from .embedder import get_embedding
from .retriever import KnowledgeRetriever

groq_client = Groq(api_key=Config.GROQ_API_KEY)
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

def cardio_consult(symptom_summary, retriever=None):
    """
    Agentic AI consult using Groq for inference, OpenAI for embeddings.
    """
    query_vec = get_embedding(symptom_summary)
    context = ""

    if retriever:
        docs = retriever.search(query_vec)
        context = "\n".join(docs)

    prompt = f"""
    You are CardioConsult, an empathetic and knowledgeable AI cardiology assistant.
    Analyze the user's symptoms and provide:
    1. Likely heart condition (if any)
    2. Possible causes
    3. Recommended lifestyle or next medical steps
    4. Urgency level
    
    Patient summary: {symptom_summary}
    Context (medical notes, RAG docs): {context}
    """

    completion = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are CardioConsult, a heart health AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=600
    )

    return completion.choices[0].message.content
