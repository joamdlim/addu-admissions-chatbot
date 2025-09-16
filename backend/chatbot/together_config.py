"""
Configuration settings for Together AI models.
Replaces llama_config.py for hosted API usage.
"""

# Together AI model configuration
TOGETHER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"  # Fast and efficient
# Alternative models you can use:
# "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"  # More powerful but slower
# "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"  # Most powerful but slowest

# Generation parameters - optimized for Together AI
GENERATION_CONFIG = {
    "max_tokens": 768,           # Cap tokens for faster replies
    "temperature": 0.7,          # Balanced creativity
    "top_p": 0.9,               # Nucleus sampling
    "top_k": 40,                # Top-k sampling
    "repetition_penalty": 1.1,   # Penalty for repeating tokens
    "stop": ["</answer>", "<|user|>", "<|system|>", "<|assistant|>"]
}

# Typo correction parameters (extremely fast)
TYPO_CORRECTION_CONFIG = {
    "max_tokens": 30,           # Minimal tokens for typo correction
    "temperature": 0.1,         # Very deterministic
    "top_p": 0.9,
    "top_k": 5,                 # Very limited token selection
}

# Word prediction parameters (extremely fast)
WORD_PREDICTION_CONFIG = {
    "max_tokens": 3,            # Just a few tokens
    "temperature": 0.7,         # Keep some creativity
    "top_p": 0.9,
    "top_k": 10,
}

# API configuration
API_CONFIG = {
    "timeout": 30,              # Request timeout in seconds
    "max_retries": 3,           # Number of retries for failed requests
    "retry_delay": 1,           # Delay between retries in seconds
}
