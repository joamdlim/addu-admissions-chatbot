# backend/chatbot/llama_interface.py
from llama_cpp import Llama
import os
from typing import Dict, List, Union, Iterator, Any, TypeVar, cast

# Import configuration - change from relative to absolute import
try:
    from llama_config import (
        MODEL_PATH,
        MODEL_CONFIG,
        GENERATION_CONFIG,
        TYPO_CORRECTION_CONFIG,
        WORD_PREDICTION_CONFIG
    )
except ImportError:
    # Fallback to direct import if run from different directory
    from chatbot.llama_config import (
        MODEL_PATH,
        MODEL_CONFIG,
        GENERATION_CONFIG,
        TYPO_CORRECTION_CONFIG,
        WORD_PREDICTION_CONFIG
    )

print("🧠 Loading model...")
try:
    # Create the model instance with optimized parameters from config
    llm = Llama(
        model_path=MODEL_PATH,
        **MODEL_CONFIG
    )
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model load failed:", e)
    raise

# Define a type variable for the llama-cpp-python return types
T = TypeVar('T')

def extract_response_text(result: Any) -> str:
    """Helper function to extract text from model response regardless of return type"""
    if hasattr(result, "__iter__") and not isinstance(result, dict):
        # For streaming responses
        response_text = ""
        for item in result:
            if isinstance(item, dict) and "choices" in item and item["choices"]:
                response_text += item["choices"][0]["text"]
        return response_text.strip()
    else:
        # For non-streaming responses (dictionary)
        if isinstance(result, dict) and "choices" in result and result["choices"]:
            return result["choices"][0]["text"].strip()
    
    # Fallback
    return ""

def correct_typos(text: str) -> str:
    """Correct typos in the input text using the Llama model"""
    # Use a more explicit prompt that asks for direct correction
    prompt = f"""
    Please correct the typos in the following text and return ONLY the corrected text without any explanations:
    
    "{text}"
    
    Corrected text:
    """
    
    # Generate response with optimized parameters
    result = llm(prompt, echo=False, **TYPO_CORRECTION_CONFIG)
    
    # Extract and return the response text
    corrected_text = extract_response_text(result)
    return corrected_text if corrected_text else text

def predict_next_words(text: str, num_suggestions: int = 3) -> List[str]:
    """Predict the next words that might follow the input text"""
    suggestions = []
    
    # Generate multiple suggestions with different temperatures
    for i in range(num_suggestions):
        # Adjust temperature for diversity
        params = WORD_PREDICTION_CONFIG.copy()
        params["temperature"] = 0.5 + (i * 0.2)
        
        # Generate response
        result = llm(text, echo=False, **params)
        
        # Extract suggestion
        suggestion = extract_response_text(result)
        if suggestion:
            suggestions.append(suggestion)
        
    return suggestions