"""
Token counting and context budgeting utilities for LLaMA-2-7B conversation memory.
"""

import os
import re
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
import logging

# Configure logging
logger = logging.getLogger(__name__)

class TokenCounter:
    """Handles accurate token counting for LLaMA-2-7B model"""
    
    def __init__(self):
        self._tokenizer = None
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load the LLaMA-2 tokenizer with fallback options"""
        try:
            # Try to load LLaMA-2 tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            logger.info("✅ Loaded LLaMA-2 tokenizer")
        except Exception as e:
            logger.warning(f"Failed to load LLaMA-2 tokenizer: {e}")
            try:
                # Fallback to a similar tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
                logger.info("✅ Loaded fallback tokenizer")
            except Exception as e2:
                logger.error(f"Failed to load any tokenizer: {e2}")
                self._tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the loaded tokenizer"""
        if not text or not isinstance(text, str):
            return 0
        
        if self._tokenizer is None:
            # Fallback: rough estimation (1 token ≈ 4 characters for LLaMA)
            return max(1, len(text) // 4)
        
        try:
            tokens = self._tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using fallback")
            # Fallback estimation
            return max(1, len(text) // 4)
    
    def estimate_tokens_fast(self, text: str) -> int:
        """Fast token estimation without loading heavy models"""
        if not text:
            return 0
        
        # LLaMA-2 rough estimation rules
        word_count = len(text.split())
        char_count = len(text)
        
        # LLaMA-2 typically uses ~1.3 tokens per word
        # But special characters and formatting affect this
        estimated = int(word_count * 1.3)
        
        # Character-based fallback (useful for very short texts)
        char_based = max(1, char_count // 4)
        
        # Return the more conservative (higher) estimate
        return max(estimated, char_based)

class ConversationTokenBudget:
    """Manages token budgeting for conversation memory strategy"""
    
    # Token budget configuration
    MAX_CONTEXT_TOKENS = 4000  # LLaMA-2-7B theoretical max
    SAFE_CONTEXT_TOKENS = 3000  # Safe working limit
    SYSTEM_PROMPT_RESERVE = 500  # Reserve for system prompt and rules
    RECENT_TURNS_RESERVE = 1500  # Reserve for recent conversation turns
    SUMMARY_RESERVE = 400  # Reserve for conversation summary
    RESPONSE_GENERATION_RESERVE = 600  # Reserve for model response generation
    
    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter
        self.system_prompt_tokens = 0
        self.summary_tokens = 0
        self.recent_turns_tokens = 0
        
    def calculate_available_tokens(self) -> Dict[str, int]:
        """Calculate available tokens for each component"""
        used_tokens = (
            self.system_prompt_tokens + 
            self.summary_tokens + 
            self.recent_turns_tokens
        )
        
        remaining = self.SAFE_CONTEXT_TOKENS - used_tokens
        
        return {
            'total_budget': self.SAFE_CONTEXT_TOKENS,
            'system_prompt': self.system_prompt_tokens,
            'summary': self.summary_tokens,
            'recent_turns': self.recent_turns_tokens,
            'used_total': used_tokens,
            'available': max(0, remaining),
            'over_budget': used_tokens > self.SAFE_CONTEXT_TOKENS,
            'response_reserve': self.RESPONSE_GENERATION_RESERVE
        }
    
    def can_add_turn(self, query_tokens: int, response_tokens: int) -> bool:
        """Check if we can add a new turn without exceeding budget"""
        budget = self.calculate_available_tokens()
        new_turn_tokens = query_tokens + response_tokens
        
        # Account for response generation reserve
        effective_available = budget['available'] - self.RESPONSE_GENERATION_RESERVE
        
        return new_turn_tokens <= effective_available
    
    def should_summarize(self, turns_count: int) -> bool:
        """Determine if summarization should be triggered"""
        budget = self.calculate_available_tokens()
        
        # Trigger summarization if:
        # 1. Over budget
        # 2. After every 10 exchanges
        # 3. Recent turns taking more than allocated space
        return (
            budget['over_budget'] or 
            turns_count >= 10 or
            self.recent_turns_tokens > self.RECENT_TURNS_RESERVE
        )
    
    def update_token_counts(self, system_prompt_tokens: int = 0, 
                           summary_tokens: int = 0, 
                           recent_turns_tokens: int = 0):
        """Update the current token counts"""
        if system_prompt_tokens > 0:
            self.system_prompt_tokens = system_prompt_tokens
        if summary_tokens > 0:
            self.summary_tokens = summary_tokens
        if recent_turns_tokens > 0:
            self.recent_turns_tokens = recent_turns_tokens

class ConversationContextBuilder:
    """Builds conversation context within token limits"""
    
    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter
        self.budget = ConversationTokenBudget(token_counter)
    
    def build_context(self, system_prompt: str, summary: str, 
                     recent_turns: List[Dict], current_query: str) -> Tuple[str, Dict]:
        """Build complete context string within token limits"""
        
        # Count tokens for each component
        system_tokens = self.token_counter.count_tokens(system_prompt)
        summary_tokens = self.token_counter.count_tokens(summary)
        query_tokens = self.token_counter.count_tokens(current_query)
        
        # Calculate recent turns tokens and potentially trim
        recent_turns_text, recent_tokens = self._optimize_recent_turns(recent_turns)
        
        # Update budget tracker
        self.budget.update_token_counts(
            system_prompt_tokens=system_tokens,
            summary_tokens=summary_tokens,
            recent_turns_tokens=recent_tokens
        )
        
        # Build final context
        context_parts = []
        
        if system_prompt:
            context_parts.append(f"<system>\n{system_prompt}\n</system>")
        
        if summary:
            context_parts.append(f"<conversation_summary>\n{summary}\n</conversation_summary>")
        
        if recent_turns_text:
            context_parts.append(f"<recent_conversation>\n{recent_turns_text}\n</recent_conversation>")
        
        context_parts.append(f"<current_query>\n{current_query}\n</current_query>")
        
        full_context = "\n\n".join(context_parts)
        
        # Get budget info
        budget_info = self.budget.calculate_available_tokens()
        budget_info['query_tokens'] = query_tokens
        budget_info['context_tokens'] = self.token_counter.count_tokens(full_context)
        
        return full_context, budget_info
    
    def _optimize_recent_turns(self, recent_turns: List[Dict]) -> Tuple[str, int]:
        """Optimize recent turns to fit within budget, trimming if necessary"""
        if not recent_turns:
            return "", 0
        
        # Convert turns to text format
        turns_text = []
        total_tokens = 0
        
        # Process turns in reverse order (most recent first)
        for turn in reversed(recent_turns):
            turn_text = f"Human: {turn.get('query', '')}\nAssistant: {turn.get('response', '')}"
            turn_tokens = self.token_counter.count_tokens(turn_text)
            
            # Check if adding this turn would exceed the recent turns budget
            if total_tokens + turn_tokens > self.budget.RECENT_TURNS_RESERVE:
                # If we have no turns yet, take at least the most recent one (truncated if needed)
                if not turns_text:
                    turn_text = self._truncate_turn(turn_text, self.budget.RECENT_TURNS_RESERVE)
                    turn_tokens = self.token_counter.count_tokens(turn_text)
                    turns_text.append(turn_text)
                    total_tokens += turn_tokens
                break
            
            turns_text.append(turn_text)
            total_tokens += turn_tokens
        
        # Reverse back to chronological order
        turns_text.reverse()
        
        return "\n\n".join(turns_text), total_tokens
    
    def _truncate_turn(self, turn_text: str, max_tokens: int) -> str:
        """Truncate a turn to fit within token limit"""
        if self.token_counter.count_tokens(turn_text) <= max_tokens:
            return turn_text
        
        # Simple truncation strategy: keep the query and truncate the response
        lines = turn_text.split('\n')
        if len(lines) >= 2:
            human_line = lines[0]  # "Human: ..."
            assistant_prefix = "Assistant: "
            
            # Reserve tokens for human query and assistant prefix
            reserved_tokens = self.token_counter.count_tokens(human_line + "\n" + assistant_prefix)
            available_for_response = max_tokens - reserved_tokens - 10  # small buffer
            
            if available_for_response > 0:
                # Estimate characters we can keep for the response
                response_text = turn_text.split("Assistant: ", 1)[1] if "Assistant: " in turn_text else ""
                
                # Rough estimation: keep about 4 characters per token
                max_chars = available_for_response * 4
                if len(response_text) > max_chars:
                    response_text = response_text[:max_chars] + "... [truncated]"
                
                return f"{human_line}\n{assistant_prefix}{response_text}"
        
        # Fallback: just truncate the whole thing
        words = turn_text.split()
        estimated_words = max_tokens // 2  # Very rough estimate
        if len(words) > estimated_words:
            return " ".join(words[:estimated_words]) + "... [truncated]"
        
        return turn_text

# Initialize global instances
_token_counter = None
_context_builder = None

def get_token_counter() -> TokenCounter:
    """Get the global token counter instance"""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter

def get_context_builder() -> ConversationContextBuilder:
    """Get the global context builder instance"""
    global _context_builder
    if _context_builder is None:
        _context_builder = ConversationContextBuilder(get_token_counter())
    return _context_builder

# Convenience functions
def count_tokens(text: str) -> int:
    """Quick function to count tokens in text"""
    return get_token_counter().count_tokens(text)

def estimate_tokens_fast(text: str) -> int:
    """Quick function to estimate tokens in text"""
    return get_token_counter().estimate_tokens_fast(text)
