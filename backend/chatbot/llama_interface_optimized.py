"""
Highly optimized Llama interface for maximum speed.
This version prioritizes speed over quality.
"""

from llama_cpp import Llama
import os
import time
from typing import Any, Dict, List, Optional, Union, Iterator, TypeVar

# Import configuration
try:
    from llama_config import (
        MODEL_PATH,
        MODEL_CONFIG,
        GENERATION_CONFIG,
        TYPO_CORRECTION_CONFIG,
        WORD_PREDICTION_CONFIG
    )
except ImportError:
    from chatbot.llama_config import (
        MODEL_PATH,
        MODEL_CONFIG,
        GENERATION_CONFIG,
        TYPO_CORRECTION_CONFIG,
        WORD_PREDICTION_CONFIG
    )

# Add additional speed optimizations
SPEED_OPTIMIZED_CONFIG = MODEL_CONFIG.copy()
SPEED_OPTIMIZED_CONFIG.update({
    "n_ctx": 1024,          # Even smaller context window
    "n_batch": 1024,       # Maximum batch size
    "n_threads": 12,       # Use more threads
    "rope_freq_base": 10000,  # Adjust RoPE frequency base
    "rope_freq_scale": 0.5,   # Adjust RoPE frequency scale
})

print("🧠 Loading model with speed optimizations...")
start_time = time.time()

try:
    # Create the model instance with optimized parameters
    llm = Llama(
        model_path=MODEL_PATH,
        **SPEED_OPTIMIZED_CONFIG
    )
    load_time = time.time() - start_time
    print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    raise

def extract_response_text(result: Any) -> str:
    """Extract text from model response regardless of return type"""
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

def generate_fast_response(prompt: str, max_tokens: int = 500, stream: bool = False) -> Union[str, Iterator[Dict[str, Any]]]:
    """Generate a response with maximum speed optimizations"""
    # Use aggressive speed settings but adjusted for more complete responses
    fast_config = {
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,         # Increased from 15 for better quality
        "repeat_penalty": 1.1,
        "mirostat_mode": 0,  # Disable mirostat for longer responses
        "stop": ["</answer>"],  # Only stop at explicit end marker
        "stream": stream     # Enable/disable streaming
    }
    
    # Time the generation
    start_time = time.time()
    
    # Generate response
    result = llm(prompt, echo=False, **fast_config)
    
    # Handle streaming vs. non-streaming
    if stream:
        # For streaming, return the iterator directly
        return result
    else:
        # For non-streaming, extract response as before
        response = extract_response_text(result)
        
        # Calculate time
        gen_time = time.time() - start_time
        tokens_per_second = max_tokens / gen_time if gen_time > 0 else 0
        
        print(f"⚡ Generated ~{len(response.split())} words in {gen_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
        
        return response

def generate_response(prompt: str, max_tokens: int = 150) -> str:
    """Generate a non-streaming response (guaranteed to return a string)"""
    result = generate_fast_response(prompt, max_tokens, stream=False)
    # This cast is safe because we explicitly set stream=False
    return str(result)

def stream_response(prompt: str, max_tokens: int = 150) -> str:
    """Generate a streaming response with real-time output"""
    # Get streaming result
    stream_result = generate_fast_response(prompt, max_tokens, stream=True)
    
    # Track the full response for return value
    full_response = ""
    
    # Time the generation
    start_time = time.time()
    first_token_time = None
    
    # Process the stream
    try:
        print("💬 Response: ", end="", flush=True)
        for chunk in stream_result:
            if isinstance(chunk, dict) and "choices" in chunk and chunk["choices"]:
                if isinstance(chunk["choices"][0], dict) and "text" in chunk["choices"][0]:
                    text_chunk = chunk["choices"][0]["text"]
                    print(text_chunk, end="", flush=True)
                    full_response += text_chunk
                    
                    # Track when first token appears
                    if not first_token_time and text_chunk.strip():
                        first_token_time = time.time()
                        first_token_latency = first_token_time - start_time
                        #if first_token_latency > 0.5:  # Only log if it took more than 0.5 seconds
                         #   print(f"\n(First token: {first_token_latency:.2f}s) ", end="", flush=True)
    except Exception as e:
        print(f"\n⚠️ Streaming error: {e}")
    
    # Calculate time
    gen_time = time.time() - start_time
    tokens_per_second = max_tokens / gen_time if gen_time > 0 else 0
    
    print(f"\n\n⚡ Generated ~{len(full_response.split())} words in {gen_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
    
    return full_response

def correct_typos(text: str) -> str:
    """Correct typos in the input text - optimized for speed"""
    # Use a minimal prompt
    prompt = f"Correct typos: \"{text}\"\nCorrected:"
    
    # Use even more aggressive settings for typo correction
    fast_config = {
        "max_tokens": 20,
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 5
    }
    
    # Generate response
    result = llm(prompt, echo=False, **fast_config)
    
    # Extract and return the response text
    corrected_text = extract_response_text(result)
    return corrected_text if corrected_text else text

def predict_next_words(text: str, num_suggestions: int = 2) -> List[str]:
    """Predict next words - optimized for speed"""
    suggestions = []
    
    # Generate fewer suggestions with minimal tokens
    for i in range(min(num_suggestions, 2)):
        # Use minimal settings
        params = {
            "max_tokens": 2,
            "temperature": 0.5 + (i * 0.3),
            "top_p": 0.9,
            "top_k": 10
        }
        
        # Generate response
        result = llm(text, echo=False, **params)
        
        # Extract suggestion
        suggestion = extract_response_text(result)
        if suggestion:
            suggestions.append(suggestion)
    
    return suggestions 