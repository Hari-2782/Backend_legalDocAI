import requests
from app.config import settings

def hf_generate(prompt: str, max_new_tokens: int = 256, temperature: float = 0.2):
    url = f"https://api-inference.huggingface.co/models/{settings.HF_MODEL}"
    headers = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature}}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, list):
            text = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            text = data.get("generated_text") or data.get("output") or str(data)
        else:
            text = str(data)
        conf = 0.9 if "not stated" not in text.lower() else 0.3
        return text, conf
    else:
        return f"(HF error {resp.status_code})", 0.0