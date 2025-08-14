import requests
import logging
from app.config import settings

logger = logging.getLogger(__name__)

def call_hf_inference(prompt: str):
    """
    Call Hugging Face inference API with better error handling and logging.
    """
    try:
        headers = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}
        payload = {
            "inputs": prompt, 
            "parameters": {
                "max_new_tokens": 512,  # Increased for better responses
                "temperature": 0.3,     # Slightly more creative
                "do_sample": True,
                "top_p": 0.9
            }
        }
        
        url = f"https://api-inference.huggingface.co/models/{settings.HF_MODEL}"
        logger.info(f"Calling HF inference API: {url}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"HF API response: {data}")
            
            if isinstance(data, list):
                answer = data[0].get("generated_text", "")
            elif isinstance(data, dict):
                answer = data.get("generated_text") or data.get("output") or str(data)
            else:
                answer = str(data)
            
            # Clean up the answer
            if answer and answer != prompt:
                # Remove the original prompt from the generated text
                if answer.startswith(prompt):
                    answer = answer[len(prompt):].strip()
                
                confidence = 0.9 if "not stated" not in answer.lower() else 0.3
                logger.info(f"Generated answer: {answer[:100]}...")
                return answer, confidence
            else:
                logger.warning("Generated answer is empty or same as prompt")
                return "Unable to generate answer from the provided context.", 0.5
                
        else:
            logger.error(f"HF API error: {resp.status_code} - {resp.text}")
            
            # Try to provide a fallback answer based on the context
            if "context" in prompt.lower():
                fallback_answer = "Based on the provided document context, I can see this is a legal contract. However, I was unable to generate a specific answer to your question. Please try rephrasing your question or check if the document contains the relevant information."
                return fallback_answer, 0.6
            
            return "I encountered an error while processing your request. Please try again.", 0.0
            
    except requests.exceptions.Timeout:
        logger.error("HF API request timed out")
        return "Request timed out. Please try again.", 0.0
    except requests.exceptions.RequestException as e:
        logger.error(f"HF API request failed: {e}")
        return "Unable to connect to the AI service. Please try again later.", 0.0
    except Exception as e:
        logger.error(f"Unexpected error in inference: {e}")
        return "An unexpected error occurred. Please try again.", 0.0

def build_rag_prompt(question: str, snippets: list) -> str:
    """
    Build a better RAG prompt for legal document analysis.
    """
    if not snippets:
        return f"Question: {question}\n\nAnswer: The document does not contain any relevant information to answer this question."
    
    # Limit context length to prevent token overflow
    max_snippet_length = 2000 // len(snippets) if len(snippets) > 0 else 2000
    truncated_snippets = []
    
    for i, snippet in enumerate(snippets):
        if len(snippet) > max_snippet_length:
            truncated_snippet = snippet[:max_snippet_length] + "..."
        else:
            truncated_snippet = snippet
        truncated_snippets.append(f"Snippet {i+1}: {truncated_snippet}")
    
    context = "\n\n".join(truncated_snippets)
    
    prompt = (
        "You are a legal document assistant. Analyze the following document snippets and answer the question.\n\n"
        "Instructions:\n"
        "- Use ONLY the information provided in the snippets\n"
        "- If the answer is not in the snippets, say 'The document does not contain this information'\n"
        "- Provide specific references to snippet numbers when possible\n"
        "- Keep your answer concise and professional\n\n"
        f"Document Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    
    return prompt
