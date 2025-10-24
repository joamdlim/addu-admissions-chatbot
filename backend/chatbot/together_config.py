"""
Configuration settings for Together AI models.
Replaces llama_config.py for hosted API usage.
"""

# Together AI model configuration - Upgraded to Llama-4-Scout for reduced hallucinations
TOGETHER_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"  # Upgraded to Llama-4-Scout for better accuracy
# Alternative models you can use:
# "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"   # Smaller, faster but less accurate
# "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"  # Previous upgrade option
# "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"  # Most powerful but slowest

# Generation parameters - optimized for accuracy and reduced hallucinations
GENERATION_CONFIG = {
    "max_tokens": 1024,          # Increased for more complete responses
    "temperature": 0.3,          # Lower temperature for factual accuracy
    "top_p": 0.8,               # More focused nucleus sampling
    "top_k": 25,                # Reduced top-k for more deterministic output
    "repetition_penalty": 1.15,  # Higher penalty for repeating tokens
    "stop": ["</answer>", "<|user|>", "<|system|>", "<|assistant|>"]
}

# Typo correction parameters (extremely fast and accurate)
TYPO_CORRECTION_CONFIG = {
    "max_tokens": 30,           # Minimal tokens for typo correction
    "temperature": 0.05,        # Even more deterministic for accuracy
    "top_p": 0.8,
    "top_k": 3,                 # Very limited token selection for precision
}

# Word prediction parameters (fast and accurate)
WORD_PREDICTION_CONFIG = {
    "max_tokens": 3,            # Just a few tokens
    "temperature": 0.4,         # Lower temperature for better predictions
    "top_p": 0.8,
    "top_k": 8,
}

# API configuration
API_CONFIG = {
    "timeout": 45,              # Increased timeout for 70B model
    "max_retries": 3,           # Number of retries for failed requests
    "retry_delay": 1,           # Delay between retries in seconds
}

# Anti-hallucination configuration profiles
ACCURACY_PROFILES = {
    "high_accuracy": {
        "temperature": 0.1,
        "top_p": 0.7,
        "top_k": 15,
        "repetition_penalty": 1.2
    },
    "balanced": {
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 25,
        "repetition_penalty": 1.15
    },
    "creative": {
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1
    }
}
