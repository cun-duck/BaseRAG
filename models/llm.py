from huggingface_hub import InferenceClient
from requests.exceptions import HTTPError  
from utils.error_handling import handle_errors
from config import app_config
import time

client = InferenceClient(
    provider="hf-inference",
    api_key=app_config.hf_api_key
)

@handle_errors
def generate_response(prompt: str, model: str, context: str = "") -> tuple[str, dict]:
    messages = [
        {
            "role": "system",
            "content": f"Gunakan konteks berikut untuk menjawab pertanyaan:\n{context}\n\n" 
                      "Jika konteks tidak relevan, jawab menggunakan pengetahuan umum."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        
        token_usage = {
            'prompt_tokens': completion.usage.prompt_tokens,
            'completion_tokens': completion.usage.completion_tokens,
            'total_tokens': completion.usage.total_tokens
        } if completion.usage else {}
        
        return str(completion.choices[0].message.content), token_usage
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return f"Error: {str(e)}", {}