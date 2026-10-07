from rag_project2 import settings
from openai import OpenAI

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def generate_answer(question, context_chunks=None, history_text=None):
    context = ""
    if context_chunks:
        context = "\n\n".join(context_chunks)

    history_text = history_text or ""

    prompt = f"""
        Use only the context below to answer the question. If the context does not contain enough information, 
        respond with: "I don't know based on the provided context.
        " Do not invent facts or use any information not present in the context.
        Respond for general greetings or non-question inputs with a simple "Hello! How can I assist you today?" without using any context information.

    Context:
    {context}


    Conversation history:
    {history_text}


    Question:
    {question}
    """

    response = client.responses.create(
        model=settings.OPENAI_MODEL,
        max_output_tokens=1500,
        input=prompt,
    )

    return response.output_text
