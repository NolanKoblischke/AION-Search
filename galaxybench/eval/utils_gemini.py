# pip install google-genai
import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

def generate(prompt: str, image_path: str, model: str, thinking_budget: int = None, precontext_parts=None):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    parts = []
    
    # Add precontext parts if provided
    if precontext_parts is not None:
        parts.extend(precontext_parts)
    
    # Add image and text parts
    parts.extend([
        types.Part.from_bytes(
            mime_type="image/png",
            data=image_bytes,
        ),
        types.Part.from_text(text=prompt),
    ])
    
    contents = [
        types.Content(
            role="user",
            parts=parts,
        )
    ]

    # Only set generate_content_config if thinking_budget is provided
    if thinking_budget is not None and thinking_budget != -1:
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=thinking_budget,
            ),
            response_mime_type="text/plain",
        )
    else:
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
    except Exception as e:
        # Re-raise with more context about the error
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            raise Exception(f"429 RESOURCE_EXHAUSTED. {error_msg}")
        else:
            raise

    # Extract response text, checking for None or empty responses
    response_text = response.text if hasattr(response, 'text') else None
    
    # Check if we got a valid response
    if response_text is None or response_text.strip() == "":
        # Check if there was a safety/filter issue
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                raise ValueError(f"Generation failed with finish_reason: {candidate.finish_reason}")
        raise ValueError("API returned empty or null response")
    
    # Safely extract token counts, defaulting to None if not available
    input_token_count = getattr(response.usage_metadata, 'prompt_token_count', None) if hasattr(response, 'usage_metadata') else None
    output_token_count = getattr(response.usage_metadata, 'candidates_token_count', None) if hasattr(response, 'usage_metadata') else None
    
    # Check if response.usage_metadata has thoughts_token_count and add it if available
    if (hasattr(response, 'usage_metadata') and 
        hasattr(response.usage_metadata, 'thoughts_token_count') and 
        response.usage_metadata.thoughts_token_count is not None and
        output_token_count is not None):
        output_token_count = output_token_count + response.usage_metadata.thoughts_token_count

    return response_text, input_token_count, output_token_count