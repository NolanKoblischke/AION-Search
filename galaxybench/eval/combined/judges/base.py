"""Base judge class for the unified evaluation framework."""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()


class BaseJudge(ABC):
    """Base class for judging galaxy descriptions."""
    
    def __init__(self, model: str = 'gpt-4.1-nano'):
        self.model = model
        # Set reasoning effort and thinking budget based on specific models
        self.reasoning_effort = "low" if model == "o4-mini" else None
        self.thinking_budget = 24000 if model == "gemini-2.5-flash-preview-05-20" else None
    
    def _is_gemini_model(self) -> bool:
        """Check if the model is a Gemini model."""
        return "gemini" in self.model.lower()
    
    def _supports_reasoning(self) -> bool:
        """Check if the model supports reasoning effort parameters."""
        if self._is_gemini_model():
            # Gemini models with thinking support
            return "gemini" in self.model
        else:
            # OpenAI reasoning models
            reasoning_models = {
                "o1", "o1-mini", "o1-preview", 
                "o3", "o3-mini", 
                "o4", "o4-mini"
            }
            
            # Check if model name matches any reasoning model
            model_lower = self.model.lower()
            for reasoning_model in reasoning_models:
                if reasoning_model in model_lower:
                    return True
            return False
    
    def _get_structured_output_openai(self, prompt: str, schema_class) -> Dict[str, Any]:
        """Get structured output from OpenAI."""
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Get the schema
        schema = schema_class.model_json_schema()
        
        # For OpenAI structured outputs, all properties must be in required array
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
        
        # Recursively add additionalProperties: false
        def add_additional_properties_false(obj):
            if isinstance(obj, dict):
                if obj.get("type") == "object":
                    obj["additionalProperties"] = False
                for value in obj.values():
                    add_additional_properties_false(value)
            elif isinstance(obj, list):
                for item in obj:
                    add_additional_properties_false(item)
        
        add_additional_properties_false(schema)
        
        # Use structured outputs
        request_kwargs = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_class.__name__.lower(),
                    "strict": True,
                    "schema": schema
                }
            }
        }
        
        # Only include reasoning parameter for models that support it
        if self._supports_reasoning() and self.reasoning_effort:
            request_kwargs["reasoning"] = {"effort": self.reasoning_effort}
        
        try:
            response = client.responses.create(**request_kwargs)
            response_text = getattr(response, 'output_text', None)
            
            # Check if we got a valid response
            if response_text is None or response_text.strip() == "":
                raise ValueError("Judge API returned empty or null response")
                
            return json.loads(response_text)
        except Exception as e:
            # Re-raise with more context about the error for quota handling
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                raise Exception(f"429 RESOURCE_EXHAUSTED. {error_msg}")
            else:
                raise
    
    def _get_structured_output_gemini(self, prompt: str, schema_class) -> Dict[str, Any]:
        """Get structured output from Gemini."""
        from google import genai
        from google.genai.types import GenerateContentConfig
        
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Build config
        config_kwargs = {
            "response_mime_type": "application/json",
            "response_schema": schema_class,
        }
        
        # Add thinking budget if supported and provided
        if self._supports_reasoning() and self.thinking_budget:
            from google.genai.types import ThinkingConfig
            config_kwargs["thinking_config"] = ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        
        config = GenerateContentConfig(**config_kwargs)
        
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            
            # Check if we got valid parsed output
            parsed_result = getattr(response, 'parsed', None)
            if parsed_result is None:
                raise ValueError("Judge API returned null parsed result")
                
            return parsed_result
        except Exception as e:
            # Re-raise with more context about the error for quota handling
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                raise Exception(f"429 RESOURCE_EXHAUSTED. {error_msg}")
            else:
                raise
    
    def _get_structured_output(self, prompt: str, schema_class) -> Dict[str, Any]:
        """Get structured output from either OpenAI or Gemini based on model type."""
        if self._is_gemini_model():
            return self._get_structured_output_gemini(prompt, schema_class)
        else:
            return self._get_structured_output_openai(prompt, schema_class)

    @abstractmethod
    def judge_description(self, description: str, galaxy_info: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Judge a galaxy description.
        
        Args:
            description: The generated description to judge
            galaxy_info: Ground truth information about the galaxy
            
        Returns:
            Tuple of (judge_results_dict, score)
            The judge_results_dict should contain all judging information
            The score should be between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def get_score_field_name(self) -> str:
        """Get the name of the score field for this judge type."""
        pass
    
    @abstractmethod
    def format_judge_results_for_display(self, judge_results: Dict[str, Any], galaxy_info: Dict[str, Any]) -> str:
        """
        Format judge results as HTML for display.
        
        Args:
            judge_results: The judging results dictionary
            galaxy_info: The galaxy information
            
        Returns:
            HTML string for displaying the judge results
        """
        pass
    
    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single record by judging its description.
        
        Args:
            record: Record containing 'result' (description) and galaxy info
            
        Returns:
            Updated record with judging results
        """
        try:
            # Skip if no description result
            if 'result' not in record or not record['result']:
                record[self.get_score_field_name()] = None
                record['judge_error'] = "No result"
                return record
            
            description = record['result']
            
            # Judge the description
            judge_results, score = self.judge_description(description, record)
            
            # Add judging results to the record
            record[self.get_score_field_name()] = score
            record['judge_results'] = judge_results
            record['judge_model'] = self.model
            if self.reasoning_effort:
                record['judge_reasoning_effort'] = self.reasoning_effort
            if self.thinking_budget:
                record['judge_thinking_budget'] = self.thinking_budget
            
            return record
            
        except Exception as e:
            record[self.get_score_field_name()] = None
            record['judge_error'] = str(e)
            return record 