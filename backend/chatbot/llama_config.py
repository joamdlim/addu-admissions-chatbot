# backend/chatbot/llama_config.py
"""
Configuration settings for the Llama model.
Adjusted for maximum speed performance.
"""

# Model path
MODEL_PATH = "D:/Joash/models/llama-2-7b-chat.Q4_K_M.gguf"

# Model loading parameters - optimized for speed
MODEL_CONFIG = {
    # Core parameters
    "n_ctx": 1024,         # Reduced context window for faster processing
    "n_threads": 12,       # Increased threads for better CPU utilization
    "n_batch": 1024,       # Larger batch size for more efficient processing
    "n_gpu_layers": 32,    # Offload more layers to GPU if available
    
    # Memory optimization
    "f16_kv": True,        # Use half-precision for key/value cache
    "use_mlock": True,     # Lock memory to prevent swapping
    "use_mmap": True,      # Use memory mapping for faster loading
    "vocab_only": False,   # Load the full model
    
    # Performance optimizations
    "seed": 42,            # Consistent seed for reproducibility
    "verbose": False,      # Disable verbose output
    "logits_all": False,   # Don't compute logits for all tokens
    "embedding": False     # Don't return embeddings
}

# Generation parameters - tuned for speed
GENERATION_CONFIG = {
    "max_tokens": 80,      # Further reduced token count
    "temperature": 0.7,    # Keep creativity balanced
    "top_p": 0.9,          # More aggressive nucleus sampling
    "top_k": 20,           # More aggressive top-k filtering
    "repeat_penalty": 1.1, # Penalty for repeating tokens
    "mirostat_mode": 2,    # Use adaptive sampling
    "mirostat_tau": 5.0,   # Target entropy
    "mirostat_eta": 0.1    # Learning rate
}

# Typo correction parameters (extremely fast)
TYPO_CORRECTION_CONFIG = {
    "max_tokens": 30,      # Minimal tokens for typo correction
    "temperature": 0.1,    # Very deterministic
    "top_p": 0.9,
    "top_k": 5,            # Very limited token selection
    "mirostat_mode": 2
}

# Word prediction parameters (extremely fast)
WORD_PREDICTION_CONFIG = {
    "max_tokens": 3,       # Just a few tokens
    "temperature": 0.7,    # Keep some creativity
    "top_p": 0.9,
    "top_k": 10,
    "mirostat_mode": 2
} 