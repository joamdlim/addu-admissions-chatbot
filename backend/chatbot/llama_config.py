# backend/chatbot/llama_config.py
"""
Configuration settings for the Llama model.
Adjusted for balanced performance across GTX 1660 Ti (6 GB) and GTX 1650 (4 GB).
"""

# Model path
MODEL_PATH = "model/llama-2-7b-chat.Q4_K_M.gguf"

# Model loading parameters - optimized for speed and stability
MODEL_CONFIG = {
    # Core parameters
    "n_ctx": 2048,         # Balanced: enough for summarization strategy
    "n_threads": 8,        # Safe for both CPUs (leaves some for OS)
    "n_batch": 256,        # Conservative batch size to avoid OOM on 4 GB VRAM
    "n_gpu_layers": 12,    # Safe for GTX 1650 (4 GB), runs fine on 1660 Ti too
    
    # Memory optimization
    "f16_kv": True,        # Use half-precision for key/value cache
    "use_mlock": True,     # Lock memory to prevent swapping
    "use_mmap": True,      # Use memory mapping for faster loading
    "vocab_only": False,   # Load the full model
    
    # Performance optimizations
    "verbose": False,      # Disable verbose output
    "logits_all": False,   # Don't compute logits for all tokens
    "embedding": False     # Don't return embeddings
}

# Generation parameters - tuned for complete but efficient responses
GENERATION_CONFIG = {
    "max_tokens": 768,     # Cap tokens for faster replies (10–20s instead of minutes)
    "temperature": 0.7,    # Balanced creativity
    "top_p": 0.9,          # Nucleus sampling
    "top_k": 40,           # Wider top-k improves coherence
    "repeat_penalty": 1.1, # Penalty for repeating tokens
    "mirostat_mode": 0,    # Disable mirostat for predictable outputs
    "mirostat_tau": 5.0,   # Default entropy target
    "mirostat_eta": 0.1,   # Default learning rate
    "seed": 42             # Moved here (correct place for reproducibility)
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
