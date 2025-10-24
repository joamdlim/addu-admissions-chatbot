"""
Together AI interface to replace local GGUF model.
Provides the same interface as llama_interface_optimized.py but uses Together AI's hosted API.
"""

import os
import time
import json
from typing import Any, Dict, List, Optional, Union, Iterator
from together import Together
import requests

# Configuration for Together AI - Upgraded to Llama-4-Scout for reduced hallucinations
TOGETHER_CONFIG = {
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",  # Upgraded to Llama-4-Scout for better accuracy
    "max_tokens": 1024,        # Increased for more complete responses
    "temperature": 0.3,        # Lower temperature to reduce hallucinations
    "top_p": 0.8,             # More focused sampling
    "top_k": 25,              # Reduced for more deterministic output
    "repetition_penalty": 1.15, # Slightly higher to avoid repetition
    "stop": ["</answer>", "<|user|>", "<|system|>", "<|assistant|>"]
}

# Typo correction parameters (extremely fast and accurate)
TYPO_CORRECTION_CONFIG = {
    "max_tokens": 30,
    "temperature": 0.05,       # Even more deterministic for typo correction
    "top_p": 0.8,
    "top_k": 3                 # Very focused for accurate corrections
}

# Word prediction parameters (fast and accurate)
WORD_PREDICTION_CONFIG = {
    "max_tokens": 3,
    "temperature": 0.4,        # Lower for more predictable suggestions
    "top_p": 0.8,
    "top_k": 8
}

print("🧠 Initializing Together AI client with Llama-4-Scout model...")
start_time = time.time()

try:
    # Initialize Together AI client
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    
    client = Together(api_key=api_key)
    init_time = time.time() - start_time
    print(f"✅ Together AI client initialized with Llama-4-Scout-17B-16E-Instruct in {init_time:.2f} seconds")
    print("🎯 Model upgraded to Llama-4-Scout for significantly reduced hallucinations and better context retrieval")
except Exception as e:
    print(f"❌ Together AI initialization failed: {e}")
    raise

def extract_response_text(result: Any) -> str:
    """Extract text from Together AI response"""
    if isinstance(result, dict):
        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"].strip()
    return ""

def generate_fast_response(prompt: str, max_tokens: int = 1024, stream: bool = False) -> Union[str, Iterator[Dict[str, Any]]]:
    """Generate a response using Together AI"""
    
    # Prepare messages for chat completion
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Use Together AI configuration
    config = TOGETHER_CONFIG.copy()
    config.update({
        "max_tokens": max_tokens,
        "stream": stream
    })
    
    # Time the generation
    start_time = time.time()
    
    try:
        if stream:
            # For streaming, we'll simulate it since Together AI streaming works differently
            response = client.chat.completions.create(
                model=config["model"],
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                repetition_penalty=config["repetition_penalty"],
                stop=config["stop"]
            )
            
            # Simulate streaming by yielding chunks
            full_response = response.choices[0].message.content
            chunk_size = 10  # Characters per chunk
            
            def stream_generator():
                for i in range(0, len(full_response), chunk_size):
                    chunk = full_response[i:i + chunk_size]
                    yield {
                        "choices": [{
                            "delta": {"content": chunk},
                            "text": chunk
                        }]
                    }
                    time.sleep(0.05)  # Small delay to simulate streaming
            
            return stream_generator()
        else:
            # Non-streaming response
            response = client.chat.completions.create(
                model=config["model"],
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                repetition_penalty=config["repetition_penalty"],
                stop=config["stop"]
            )
            
            response_text = response.choices[0].message.content
            
            # Calculate time
            gen_time = time.time() - start_time
            tokens_per_second = max_tokens / gen_time if gen_time > 0 else 0
            
            print(f"⚡ Generated ~{len(response_text.split())} words in {gen_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
            
            return response_text
            
    except Exception as e:
        print(f"❌ Together AI generation error: {e}")
        return ""

def generate_response(prompt: str, max_tokens: int = 1024) -> str:
    """Generate a non-streaming response (guaranteed to return a string)"""
    result = generate_fast_response(prompt, max_tokens, stream=False)
    return str(result)

def stream_response(prompt: str, max_tokens: int = 1024) -> str:
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
        print("�� Response: ", end="", flush=True)
        for chunk in stream_result:
            if isinstance(chunk, dict) and "choices" in chunk and chunk["choices"]:
                if isinstance(chunk["choices"][0], dict) and "delta" in chunk["choices"][0]:
                    text_chunk = chunk["choices"][0]["delta"]["content"]
                    print(text_chunk, end="", flush=True)
                    full_response += text_chunk
                    
                    # Track when first token appears
                    if not first_token_time and text_chunk.strip():
                        first_token_time = time.time()
                        first_token_latency = first_token_time - start_time
                        
    except Exception as e:
        print(f"\n⚠️ Streaming error: {e}")
    
    # Calculate time
    gen_time = time.time() - start_time
    tokens_per_second = max_tokens / gen_time if gen_time > 0 else 0
    
    print(f"\n\n⚡ Generated ~{len(full_response.split())} words in {gen_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
    
    return full_response

def correct_typos(text: str) -> str:
    """Correct typos in the input text using Together AI"""
    prompt = f"Correct typos: \"{text}\"\nCorrected:"
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model=TOGETHER_CONFIG["model"],
            messages=messages,
            max_tokens=TYPO_CORRECTION_CONFIG["max_tokens"],
            temperature=TYPO_CORRECTION_CONFIG["temperature"],
            top_p=TYPO_CORRECTION_CONFIG["top_p"],
            top_k=TYPO_CORRECTION_CONFIG["top_k"]
        )
        
        corrected_text = response.choices[0].message.content.strip()
        return corrected_text if corrected_text else text
        
    except Exception as e:
        print(f"⚠️ Typo correction failed: {e}")
        return text

def predict_next_words(text: str, num_suggestions: int = 2) -> List[str]:
    """Predict next words using Together AI"""
    suggestions = []
    
    try:
        for i in range(min(num_suggestions, 2)):
            messages = [{"role": "user", "content": f"Complete this text: {text}"}]
            
            response = client.chat.completions.create(
                model=TOGETHER_CONFIG["model"],
                messages=messages,
                max_tokens=WORD_PREDICTION_CONFIG["max_tokens"],
                temperature=WORD_PREDICTION_CONFIG["temperature"] + (i * 0.3),
                top_p=WORD_PREDICTION_CONFIG["top_p"],
                top_k=WORD_PREDICTION_CONFIG["top_k"]
            )
            
            suggestion = response.choices[0].message.content.strip()
            if suggestion:
                suggestions.append(suggestion)
                
    except Exception as e:
        print(f"⚠️ Word prediction failed: {e}")
    
    return suggestions

# Create a mock llm object for compatibility with existing code
class MockLLM:
    """Mock LLM object to maintain compatibility with existing llama-cpp-python code"""
    
    def __call__(self, prompt: str, echo: bool = False, **kwargs) -> Union[str, Iterator[Dict[str, Any]]]:
        """Call the Together AI interface with the same signature as llama-cpp-python"""
        max_tokens = kwargs.get("max_tokens", 150)
        stream = kwargs.get("stream", False)
        
        if stream:
            return generate_fast_response(prompt, max_tokens, stream=True)
        else:
            response_text = generate_fast_response(prompt, max_tokens, stream=False)
            # Return in llama-cpp-python format
            return {
                "choices": [{
                    "text": response_text
                }]
            }

# Create the mock llm instance
llm = MockLLM()

def stream_response_together(prompt: str, max_tokens: int = 1024):
    """
    Generate a streaming response that yields string chunks
    This is for the chatbot's streaming interface
    """
    # Get streaming result
    stream_result = generate_fast_response(prompt, max_tokens, stream=True)
    
    # Process the stream and yield string chunks
    try:
        for chunk in stream_result:
            if isinstance(chunk, dict) and "choices" in chunk and chunk["choices"]:
                if isinstance(chunk["choices"][0], dict) and "delta" in chunk["choices"][0]:
                    text_chunk = chunk["choices"][0]["delta"]["content"]
                    yield text_chunk  # Yield just the string, not the dict
    except Exception as e:
        print(f"⚠️ Streaming error: {e}")
        yield ""  # Yield empty string on error

# Export the same interface as the original file
__all__ = [
    "generate_fast_response",
    "generate_response", 
    "stream_response",
    "stream_response_together",  # Add this
    "correct_typos",
    "predict_next_words",
    "llm",
    "TOGETHER_CONFIG",
    "TYPO_CORRECTION_CONFIG", 
    "WORD_PREDICTION_CONFIG"
]
