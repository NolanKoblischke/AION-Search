from openai import OpenAI
import base64
import os

from dotenv import load_dotenv
load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate(prompt: str, image_path: str, model: str, reasoning_effort: str = None) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    base64_image = encode_image(image_path)

    request_kwargs = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                    {
                        "type": "input_text",
                        "text": prompt
                    }
                ]
            }
        ],
        "text": {
            "format": {
                "type": "text"
            }
        },
        "tools": [],
        "store": True
    }

    if reasoning_effort is not None:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**request_kwargs)
    
    # Extract response text, checking for None or empty responses
    response_text = getattr(response, 'output_text', None)
    
    # Check if we got a valid response
    if response_text is None or response_text.strip() == "":
        raise ValueError("API returned empty or null response")
    
    # Safely extract token counts, defaulting to None if not available
    input_token_count = getattr(response.usage, 'input_tokens', None) if hasattr(response, 'usage') else None
    # OpenAI already includes reasoning tokens in the output token count
    output_token_count = getattr(response.usage, 'output_tokens', None) if hasattr(response, 'usage') else None

    return response_text, input_token_count, output_token_count
