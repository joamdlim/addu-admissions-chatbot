"""
Script to run the chatbot with different configurations.
"""

import argparse
import sys
import os

def main():
    """Main entry point for the chatbot runner"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the ADDU chatbot with different configurations")
    parser.add_argument("--mode", type=str, choices=["original", "fast", "hybrid"], default="hybrid",
                        help="Chatbot mode: original (standard), fast (speed-optimized), hybrid (balanced)")
    parser.add_argument("--no-typo-correction", action="store_true",
                        help="Disable typo correction for faster response")
    parser.add_argument("--max-tokens", type=int, default=60,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--top-k", type=int, default=2,
                        help="Number of documents to retrieve")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming responses")
    
    args = parser.parse_args()
    
    # Import the appropriate chatbot based on mode
    if args.mode == "original":
        print("🤖 Running original chatbot...")
        from test_hybrid_chatbot import test_hybrid_chatbot_interactive
        test_hybrid_chatbot_interactive()
    elif args.mode == "fast":
        print("⚡ Running fast optimized chatbot...")
        from test_optimized_chatbot import test_optimized_chatbot
        test_optimized_chatbot()
    else:  # hybrid mode
        print("⚡🔍 Running fast hybrid chatbot...")
        from fast_hybrid_chatbot import FastHybridChatbot
        
        # Create and run the fast hybrid chatbot
        chatbot = FastHybridChatbot()
        
        print("\n⚡🔍 FAST HYBRID CHATBOT MODE ⚡🔍")
        print("Type 'exit' to quit\n")
        
        while True:
            query = input("🔍 Query: ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            # Process query with command-line options
            response, relevant_docs = chatbot.process_query(
                query, 
                correct_spelling=not args.no_typo_correction,
                max_tokens=args.max_tokens,
                stream=not args.no_stream
            )
            
            # Show document info
            print(f"\n📚 Retrieved {len(relevant_docs)} document(s)")
            for i, doc in enumerate(relevant_docs):
                print(f"  [{i+1}] {doc['id']} (Relevance: {doc['relevance']:.4f})")
            
            # Show response
            print("\n💬 Response: ", end="")
            print(response)
            
            print("\n")
            print("-" * 80)

if __name__ == "__main__":
    main() 