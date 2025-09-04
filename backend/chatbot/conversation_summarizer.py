"""
Conversation summarization logic for managing long conversations within token limits.
Uses both rule-based and LLM-based summarization strategies.
"""

import re
from typing import Dict, List, Optional, Tuple
from django.utils import timezone
from .token_utils import get_token_counter, get_context_builder
from .models import Conversation, ConversationTurn, ConversationSummary
import logging

logger = logging.getLogger(__name__)

class ConversationSummarizer:
    """Handles conversation summarization with multiple strategies"""
    
    def __init__(self):
        self.token_counter = get_token_counter()
        self.context_builder = get_context_builder()
        
        # Summarization prompts
        self.summarization_prompt = """<system>
You are an expert at creating concise conversation summaries. Your task is to summarize the provided conversation turns while preserving all important information.

GUIDELINES:
- Keep key facts, user questions, and important answers
- Preserve specific details mentioned (names, dates, numbers, requirements)
- Maintain the logical flow of the conversation
- Focus on information that might be referenced later
- Use clear, factual language
- Aim for 200-400 tokens maximum

AVOID:
- Repetitive information
- Casual conversation elements
- Exact wording (paraphrase instead)
- Generic responses that don't add value
</system>

<conversation_to_summarize>
{conversation_turns}
</conversation_to_summarize>

Create a concise summary that preserves all important information from this conversation:"""

        self.recursive_summarization_prompt = """<system>
You are summarizing conversation summaries to create a more concise overview. 
Combine the provided summaries while keeping all essential information.

FOCUS ON:
- Key facts and decisions mentioned across summaries
- Important user preferences or requirements
- Specific details that might be referenced later
- Main topics and their outcomes

Aim for 150-300 tokens maximum.
</system>

<summaries_to_combine>
{previous_summaries}
</summaries_to_combine>

Create a consolidated summary that preserves the essential information:"""
    
    def should_summarize(self, conversation: Conversation) -> bool:
        """Determine if conversation needs summarization"""
        return self.context_builder.budget.should_summarize(conversation.total_exchanges)
    
    def summarize_conversation_turns(self, conversation: Conversation, 
                                   turns_to_summarize: List[ConversationTurn],
                                   use_llm: bool = True) -> ConversationSummary:
        """Summarize a range of conversation turns"""
        
        if not turns_to_summarize:
            raise ValueError("No turns provided for summarization")
        
        # Sort turns by turn number
        turns_to_summarize = sorted(turns_to_summarize, key=lambda x: x.turn_number)
        
        # Create conversation text
        conversation_text = self._format_turns_for_summarization(turns_to_summarize)
        
        # Generate summary
        if use_llm:
            summary_text = self._llm_summarize(conversation_text)
        else:
            summary_text = self._rule_based_summarize(turns_to_summarize)
        
        # Count tokens in summary
        summary_tokens = self.token_counter.count_tokens(summary_text)
        
        # Create summary record
        summary = ConversationSummary.objects.create(
            conversation=conversation,
            summary_text=summary_text,
            covers_turns_start=turns_to_summarize[0].turn_number,
            covers_turns_end=turns_to_summarize[-1].turn_number,
            summary_tokens=summary_tokens,
            summary_level=1,
            is_active=True
        )
        
        # Mark summarized turns as not in active context
        for turn in turns_to_summarize:
            turn.used_in_context = False
            turn.save()
        
        logger.info(f"✅ Created summary for turns {summary.covers_turns_start}-{summary.covers_turns_end}: {summary_tokens} tokens")
        
        return summary
    
    def _format_turns_for_summarization(self, turns: List[ConversationTurn]) -> str:
        """Format conversation turns for summarization"""
        formatted_turns = []
        
        for turn in turns:
            formatted_turns.append(f"Turn {turn.turn_number}:")
            formatted_turns.append(f"User: {turn.user_query}")
            formatted_turns.append(f"Assistant: {turn.bot_response}")
            formatted_turns.append("")  # Empty line for separation
        
        return "\n".join(formatted_turns)
    
    def _llm_summarize(self, conversation_text: str) -> str:
        """Use LLM to create conversation summary"""
        try:
            # Import LLM interface
            from .llama_interface_optimized import generate_fast_response
            
            # Create summarization prompt
            prompt = self.summarization_prompt.format(
                conversation_turns=conversation_text
            )
            
            # Generate summary with conservative token limit
            summary = generate_fast_response(
                prompt=prompt,
                max_tokens=400,  # Conservative limit for summaries
                stream=False
            )
            
            # Clean up the response
            summary = self._clean_summary_response(summary)
            
            # Validate summary length
            if self.token_counter.count_tokens(summary) > 500:
                logger.warning("LLM summary too long, falling back to rule-based")
                return self._rule_based_summarize_from_text(conversation_text)
            
            return summary
            
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}, falling back to rule-based")
            return self._rule_based_summarize_from_text(conversation_text)
    
    def _rule_based_summarize(self, turns: List[ConversationTurn]) -> str:
        """Create rule-based summary of conversation turns"""
        
        # Extract key information
        topics = []
        important_facts = []
        user_questions = []
        key_responses = []
        
        for turn in turns:
            # Extract user questions/topics
            query = turn.user_query.strip()
            if len(query) > 10:  # Ignore very short queries
                # Extract question-like patterns
                if any(word in query.lower() for word in ['what', 'how', 'why', 'when', 'where', 'can', 'should', 'do']):
                    user_questions.append(f"Asked about: {query[:100]}")
                else:
                    topics.append(query[:80])
            
            # Extract important facts from responses
            response = turn.bot_response.strip()
            facts = self._extract_important_facts(response)
            important_facts.extend(facts)
        
        # Build summary
        summary_parts = []
        
        if topics:
            summary_parts.append(f"Topics discussed: {', '.join(set(topics[:5]))}")
        
        if user_questions:
            summary_parts.append(f"Key questions: {' | '.join(user_questions[:3])}")
        
        if important_facts:
            summary_parts.append(f"Important information: {' | '.join(important_facts[:5])}")
        
        # Add conversation flow
        summary_parts.append(f"Conversation covered {len(turns)} exchanges from turn {turns[0].turn_number} to {turns[-1].turn_number}")
        
        return ". ".join(summary_parts) + "."
    
    def _rule_based_summarize_from_text(self, conversation_text: str) -> str:
        """Rule-based summarization from formatted text"""
        lines = conversation_text.split('\n')
        topics = []
        facts = []
        
        for line in lines:
            if line.startswith('User:'):
                query = line[5:].strip()
                if len(query) > 10:
                    topics.append(query[:80])
            elif line.startswith('Assistant:'):
                response = line[10:].strip()
                facts.extend(self._extract_important_facts(response))
        
        summary_parts = []
        if topics:
            summary_parts.append(f"Discussed: {', '.join(set(topics[:4]))}")
        if facts:
            summary_parts.append(f"Key points: {' | '.join(facts[:4])}")
        
        return ". ".join(summary_parts) + "."
    
    def _extract_important_facts(self, text: str) -> List[str]:
        """Extract important facts from response text"""
        facts = []
        
        # Look for factual patterns
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 150:
                continue
            
            # Patterns that indicate important information
            important_patterns = [
                r'\b(requirement|deadline|cost|price|fee)\b',
                r'\b(must|required|need to|have to)\b',
                r'\b(\d+)\s*(year|month|day|hour|percent|%|\$)\b',
                r'\b(contact|email|phone|address)\b',
                r'\b(available|offered|include)\b',
            ]
            
            for pattern in important_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    facts.append(sentence[:100])
                    break
        
        return facts[:3]  # Limit to 3 facts per response
    
    def _clean_summary_response(self, summary: str) -> str:
        """Clean up LLM-generated summary"""
        # Remove common LLM artifacts
        summary = re.sub(r'^(Summary:|Here\'s a summary:|The conversation|This conversation)', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'(</answer>|<answer>)', '', summary)
        
        # Clean whitespace
        summary = re.sub(r'\s+', ' ', summary)
        summary = summary.strip()
        
        # Ensure it doesn't start with lowercase after cleaning
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        return summary
    
    def recursive_summarize(self, conversation: Conversation, 
                          summaries_to_combine: List[ConversationSummary]) -> ConversationSummary:
        """Combine multiple summaries into a single, more concise summary"""
        
        if len(summaries_to_combine) < 2:
            raise ValueError("Need at least 2 summaries to combine")
        
        # Sort summaries by turn range
        summaries_to_combine = sorted(summaries_to_combine, key=lambda x: x.covers_turns_start)
        
        # Create combined summary text
        summary_texts = [f"Summary {i+1}: {s.summary_text}" for i, s in enumerate(summaries_to_combine)]
        combined_text = "\n\n".join(summary_texts)
        
        # Generate recursive summary
        try:
            from .llama_interface_optimized import generate_fast_response
            
            prompt = self.recursive_summarization_prompt.format(
                previous_summaries=combined_text
            )
            
            new_summary_text = generate_fast_response(
                prompt=prompt,
                max_tokens=300,
                stream=False
            )
            
            new_summary_text = self._clean_summary_response(new_summary_text)
            
        except Exception as e:
            logger.error(f"Recursive LLM summarization failed: {e}, using rule-based")
            # Fallback: simple concatenation with pruning
            all_text = " ".join([s.summary_text for s in summaries_to_combine])
            new_summary_text = all_text[:800] + "..." if len(all_text) > 800 else all_text
        
        # Count tokens
        summary_tokens = self.token_counter.count_tokens(new_summary_text)
        
        # Create new summary
        new_summary = ConversationSummary.objects.create(
            conversation=conversation,
            summary_text=new_summary_text,
            covers_turns_start=summaries_to_combine[0].covers_turns_start,
            covers_turns_end=summaries_to_combine[-1].covers_turns_end,
            summary_tokens=summary_tokens,
            summary_level=max([s.summary_level for s in summaries_to_combine]) + 1,
            is_active=True
        )
        
        # Mark old summaries as inactive
        for summary in summaries_to_combine:
            summary.is_active = False
            summary.save()
        
        logger.info(f"✅ Created recursive summary: level {new_summary.summary_level}, {summary_tokens} tokens")
        
        return new_summary
    
    def get_active_summary(self, conversation: Conversation) -> Optional[str]:
        """Get the current active summary for a conversation"""
        active_summary = ConversationSummary.objects.filter(
            conversation=conversation,
            is_active=True
        ).order_by('-summary_level', '-created_at').first()
        
        return active_summary.summary_text if active_summary else ""
    
    def process_conversation_for_context(self, conversation: Conversation, 
                                       max_recent_turns: int = 5) -> Tuple[str, List[Dict]]:
        """Get summary and recent turns for building context"""
        
        # Get active summary
        summary = self.get_active_summary(conversation)
        
        # Get recent turns that are still in active context
        recent_turns = ConversationTurn.objects.filter(
            conversation=conversation,
            used_in_context=True
        ).order_by('-turn_number')[:max_recent_turns]
        
        # Convert to format expected by context builder
        recent_turns_data = []
        for turn in reversed(recent_turns):  # Reverse to get chronological order
            recent_turns_data.append({
                'query': turn.user_query,
                'response': turn.bot_response
            })
        
        return summary, recent_turns_data

# Global instance
_summarizer = None

def get_conversation_summarizer() -> ConversationSummarizer:
    """Get the global conversation summarizer instance"""
    global _summarizer
    if _summarizer is None:
        _summarizer = ConversationSummarizer()
    return _summarizer
