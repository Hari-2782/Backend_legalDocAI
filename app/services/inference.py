# app/services/inference.py
import requests
import logging
from app.config import settings  # OPENROUTER_API_KEY must be set in settings

logger = logging.getLogger(__name__)


def call_openrouter_inference(prompt: str):
    try:
        api_key = settings.OPENROUTER_API_KEY
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY in configuration")

        logger.info("Calling OpenRouter with model anthropic/claude-3-haiku")

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": "anthropic/claude-3-haiku",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )
        response.raise_for_status()

        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        confidence = 0.95

        lowered = answer.lower()
        if ("not stated" in lowered) or ("does not contain" in lowered):
            confidence = 0.5

        return answer, confidence

    except Exception as e:
        logger.error(f"OpenRouter API request failed: {e}")
        return "An error occurred while processing your request.", 0.0


# Backward-compatible alias for previous import name
def call_hf_inference(prompt: str):
    return call_openrouter_inference(prompt)


def build_rag_prompt(question: str, snippets: list) -> str:
    context = "\n\n".join(snippets)
    prompt = (
        "Based ONLY on the following context from a legal document, please perform the following tasks:\n\n"
        f"CONTEXT:\n---\n{context}\n---\n\n"
        f"USER'S QUESTION: \"{question}\"\n\n"
        "TASKS:\n"
        "1.  Direct Answer: Answer the user's question directly. If the answer isn't in the context, state 'The document does not provide an answer to this question.'\n"
        "2.  Summary: Provide a brief, simple-language summary of the provided context.\n"
        "3.  Key Clauses & Obligations: Identify and list the most important clauses, obligations, or deadlines mentioned.\n"
        "4.  Red Flags & Risks: Point out any potential red flags, risks, penalties, or unusual terms for the user.\n\n"
        "Please format your response clearly using markdown."
    )
    return prompt