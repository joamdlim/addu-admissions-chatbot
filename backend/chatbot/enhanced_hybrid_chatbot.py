"""
Enhanced Fast Hybrid Chatbot with conversation memory and intelligent summarization.
Integrates with the conversation memory system for long-term dialogue support.
"""

import os
import time
import uuid
from typing import Dict, List, Optional, Tuple, Generator
from django.utils import timezone

# Import existing components
from .fast_hybrid_chatbot import FastHybridChatbot
from .token_utils import get_token_counter, get_context_builder
from .conversation_summarizer import get_conversation_summarizer
from .models import Conversation, ConversationTurn, SystemPrompt

class EnhancedHybridChatbot:
    """Enhanced chatbot with conversation memory and intelligent context management"""
    
    def __init__(self, use_chroma: bool = True, chroma_collection_name: Optional[str] = None):
        # Initialize the base chatbot
        self.base_chatbot = FastHybridChatbot(
            use_chroma=use_chroma, 
            chroma_collection_name=chroma_collection_name
        )
        
        # Initialize conversation memory components
        self.token_counter = get_token_counter()
        self.context_builder = get_context_builder()
        self.summarizer = get_conversation_summarizer()
        
        # System prompt
        self.system_prompt = self._get_system_prompt()
        
        print("✅ Enhanced Hybrid Chatbot initialized with conversation memory")
    
    def _get_system_prompt(self) -> str:
        """Get the active system prompt from database or default"""
        try:
            active_prompt = SystemPrompt.objects.filter(is_active=True).first()
            if active_prompt:
                return active_prompt.prompt_text
        except Exception as e:
            print(f"⚠️ Could not load system prompt from database: {e}")
        
        # Default system prompt
        return """You are a helpful AI assistant for Ateneo de Davao University (AdDU) admissions. 
You provide accurate information about university programs, requirements, and procedures.
Always be helpful, professional, and refer to official university resources when appropriate.
If you don't know something specific, acknowledge this and suggest contacting the admissions office directly."""
    
    def get_or_create_conversation(self, session_id: str = None) -> Conversation:
        """Get existing conversation or create a new one"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        conversation, created = Conversation.objects.get_or_create(
            session_id=session_id,
            defaults={
                'title': f"Conversation {session_id[:8]}",
                'is_active': True,
                'total_exchanges': 0,
                'current_token_count': 0,
                'max_token_budget': 3000
            }
        )
        
        if created:
            print(f"📝 Created new conversation: {session_id}")
        else:
            print(f"📖 Retrieved existing conversation: {session_id} ({conversation.total_exchanges} exchanges)")
        
        return conversation
    
    def should_trigger_summarization(self, conversation: Conversation) -> bool:
        """Check if we should trigger summarization"""
        # Check if summarizer recommends summarization
        if self.summarizer.should_summarize(conversation):
            return True
        
        # Check if we have too many unsummarized turns
        active_turns = ConversationTurn.objects.filter(
            conversation=conversation,
            used_in_context=True
        ).count()
        
        return active_turns >= 10
    
    def perform_summarization(self, conversation: Conversation) -> None:
        """Perform conversation summarization when needed"""
        try:
            # Get turns that need summarization (exclude recent ones)
            turns_to_summarize = ConversationTurn.objects.filter(
                conversation=conversation,
                used_in_context=True
            ).order_by('turn_number')[:-5]  # Keep last 5 turns active
            
            if len(turns_to_summarize) >= 3:  # Only summarize if we have enough content
                print(f"🔄 Summarizing {len(turns_to_summarize)} conversation turns...")
                
                # Create summary
                summary = self.summarizer.summarize_conversation_turns(
                    conversation=conversation,
                    turns_to_summarize=list(turns_to_summarize),
                    use_llm=True
                )
                
                print(f"✅ Created summary: {summary.summary_tokens} tokens")
                
                # Check if we need recursive summarization
                all_summaries = conversation.summaries.filter(is_active=True)
                if all_summaries.count() > 3:  # If we have too many summaries
                    print("🔄 Performing recursive summarization...")
                    oldest_summaries = list(all_summaries.order_by('created_at')[:2])
                    self.summarizer.recursive_summarize(conversation, oldest_summaries)
                    print("✅ Recursive summarization completed")
        
        except Exception as e:
            print(f"⚠️ Summarization failed: {e}")
    
    def build_conversation_context(self, conversation: Conversation, current_query: str) -> str:
        """Build the complete conversation context within token limits"""
        
        # Get summary and recent turns
        summary, recent_turns = self.summarizer.process_conversation_for_context(conversation)
        
        # Build context using the context builder
        full_context, budget_info = self.context_builder.build_context(
            system_prompt=self.system_prompt,
            summary=summary,
            recent_turns=recent_turns,
            current_query=current_query
        )
        
        # Log budget information
        print(f"📊 Token Budget: {budget_info['used_total']}/{budget_info['total_budget']} tokens")
        print(f"   System: {budget_info['system_prompt']}, Summary: {budget_info['summary']}, Recent: {budget_info['recent_turns']}")
        
        if budget_info['over_budget']:
            print("⚠️ Context is over budget - consider additional summarization")
        
        return full_context
    
    def save_conversation_turn(self, conversation: Conversation, user_query: str, bot_response: str) -> ConversationTurn:
        """Save a conversation turn to the database"""
        
        # Count tokens
        query_tokens = self.token_counter.count_tokens(user_query)
        response_tokens = self.token_counter.count_tokens(bot_response)
        total_tokens = query_tokens + response_tokens
        
        # Create conversation turn
        turn = ConversationTurn.objects.create(
            conversation=conversation,
            turn_number=conversation.total_exchanges + 1,
            user_query=user_query,
            bot_response=bot_response,
            query_tokens=query_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens,
            used_in_context=True
        )
        
        # Update conversation stats
        conversation.total_exchanges += 1
        conversation.current_token_count += total_tokens
        conversation.updated_at = timezone.now()
        conversation.save()
        
        print(f"💾 Saved turn {turn.turn_number}: {query_tokens}+{response_tokens}={total_tokens} tokens")
        
        return turn
    
    def process_query_with_memory(self, query: str, session_id: str = None, 
                                 max_tokens: int = 3000, stream: bool = True,
                                 require_context: bool = True, 
                                 min_relevance: float = 0.1) -> Generator[Dict, None, None]:
        """Process query with full conversation memory support"""
        
        start_time = time.time()
        
        try:
            # Get or create conversation
            conversation = self.get_or_create_conversation(session_id)
            
            # Check if we need summarization before processing
            if self.should_trigger_summarization(conversation):
                yield {"info": "Optimizing conversation memory..."}
                self.perform_summarization(conversation)
            
            # Build conversation context
            yield {"info": "Building conversation context..."}
            conversation_context = self.build_conversation_context(conversation, query)
            
            # Get relevant documents using the base chatbot
            yield {"info": "Retrieving relevant information..."}
            
            if self.base_chatbot.use_chroma:
                retrieved_docs = self.base_chatbot._retrieve_from_chroma(query, top_k=3)
                print(f"🔍 ChromaDB Retrieved {len(retrieved_docs)} documents:")
                for i, doc in enumerate(retrieved_docs):
                    print(f"   Doc {i+1}: Relevance={doc.get('relevance', 0):.3f}, Content='{doc.get('content', '')[:100]}...'")
            else:
                retrieved_docs = self.base_chatbot._keyword_search(query, max_docs=3)
                print(f"🔍 Keyword Search Retrieved {len(retrieved_docs)} documents:")
                for i, doc in enumerate(retrieved_docs):
                    print(f"   Doc {i+1}: Relevance={doc.get('relevance', 0):.3f}, Content='{doc.get('content', '')[:100]}...'")
            
            # Filter by relevance
            relevant_docs = [doc for doc in retrieved_docs if doc.get('relevance', 0) >= min_relevance]
            print(f"📊 Relevance Filter: {len(relevant_docs)}/{len(retrieved_docs)} docs passed (min_relevance={min_relevance})")
            
            if require_context and not relevant_docs:
                print("❌ No relevant documents found - returning error")
                yield {"error": "No relevant information found in the knowledge base."}
                return
            
            # Build enhanced prompt with conversation context
            context_text = ""
            if relevant_docs:
                context_text = "\n".join([f"Context {i+1}: {doc['content'][:500]}..." 
                                        for i, doc in enumerate(relevant_docs)])
                print(f"📝 Built context from {len(relevant_docs)} documents ({len(context_text)} chars)")
            else:
                print("📝 No context text - proceeding without specific knowledge base context")
            
            # Create the complete prompt
            enhanced_prompt = f"""{conversation_context}

<knowledge_base_context>
{context_text if context_text else "No specific knowledge base context available."}
</knowledge_base_context>

<answer>
Please provide a helpful response based on the conversation context and available information:"""
            
            # Generate response using the base chatbot's LLM
            yield {"info": "Generating response..."}
            
            full_response = ""
            
            # Use the base chatbot's streaming generation
            from .llama_interface_optimized import llm
            
            stream_config = {
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "stop": ["</answer>"],  # Only stop at the end marker
                "stream": True
            }
            
            print(f"🔍 DEBUG: Stream config: {stream_config}")
            
            try:
                chunk_count = 0
                for chunk in llm(enhanced_prompt, echo=False, **stream_config):
                    chunk_count += 1
                    print(f"🔍 DEBUG: Chunk {chunk_count}: {chunk}")
                    
                    if isinstance(chunk, dict):
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            choice = chunk['choices'][0]
                            
                            text_chunk = ""
                            if 'delta' in choice and 'content' in choice['delta']:
                                text_chunk = choice['delta']['content']
                            elif 'text' in choice:
                                text_chunk = choice['text']
                            
                            if text_chunk:
                                print(f"🔍 DEBUG: Got text chunk: '{text_chunk}'")
                                full_response += text_chunk
                                if stream:
                                    yield {"chunk": text_chunk}
                
                print(f"🔍 DEBUG: Final response length: {len(full_response)}")
                print(f"🔍 DEBUG: Final response: '{full_response[:200]}...'")
                
                # Clean up response
                full_response = self._clean_response(full_response)
                
                print(f"🔍 DEBUG: Cleaned response: '{full_response[:200]}...'")
                
                # Save the conversation turn
                self.save_conversation_turn(conversation, query, full_response)
                
                # Calculate total time
                total_time = time.time() - start_time
                yield {"info": f"Total processing time: {total_time:.2f}s"}
                yield {"session_id": conversation.session_id}
                yield {"done": True}
                
            except Exception as e:
                yield {"error": f"Response generation error: {e}"}
                
        except Exception as e:
            yield {"error": f"Processing error: {e}"}
    
    def _clean_response(self, response: str) -> str:
        """Clean up the generated response"""
        # Remove system artifacts
        response = response.replace("<answer>", "").replace("</answer>", "")
        response = response.replace("<current_query>", "").replace("</current_query>", "")
        
        # Clean extra whitespace
        lines = response.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        
        return '\n\n'.join(cleaned_lines)
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for display"""
        try:
            conversation = Conversation.objects.get(session_id=session_id)
            turns = ConversationTurn.objects.filter(
                conversation=conversation
            ).order_by('-turn_number')[:limit]
            
            history = []
            for turn in reversed(turns):
                history.append({
                    'turn_number': turn.turn_number,
                    'query': turn.user_query,
                    'response': turn.bot_response,
                    'timestamp': turn.timestamp.isoformat(),
                    'tokens': turn.total_tokens
                })
            
            return history
            
        except Conversation.DoesNotExist:
            return []
    
    def clear_conversation(self, session_id: str) -> bool:
        """Clear/reset a conversation"""
        try:
            conversation = Conversation.objects.get(session_id=session_id)
            
            # Mark conversation as inactive
            conversation.is_active = False
            conversation.save()
            
            print(f"🗑️ Cleared conversation: {session_id}")
            return True
            
        except Conversation.DoesNotExist:
            return False
    
    def get_conversation_stats(self, session_id: str) -> Dict:
        """Get conversation statistics"""
        try:
            conversation = Conversation.objects.get(session_id=session_id)
            
            return {
                'session_id': session_id,
                'total_exchanges': conversation.total_exchanges,
                'created_at': conversation.created_at.isoformat(),
                'updated_at': conversation.updated_at.isoformat(),
                'total_tokens': conversation.current_token_count,
                'active_summaries': conversation.summaries.filter(is_active=True).count(),
                'is_active': conversation.is_active
            }
            
        except Conversation.DoesNotExist:
            return {}

# Global instance
_enhanced_chatbot = None

def get_enhanced_chatbot() -> EnhancedHybridChatbot:
    """Get the global enhanced chatbot instance"""
    global _enhanced_chatbot
    if _enhanced_chatbot is None:
        _enhanced_chatbot = EnhancedHybridChatbot(use_chroma=True)
    return _enhanced_chatbot
