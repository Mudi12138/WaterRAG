#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WaterRAG Interactive App
------------------------------------------
A simple command-line interface for:
  ✔ RAG Question Answering
  ✔ Literature Review Generation
  ✔ Viewing retrieved sources
  ✔ Switching LLM models
------------------------------------------
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modules
from rag_llm_reranker_simplified import RAGSystem
from iterative_review_simplified import IterativeReviewSystem

WELCOME = """
========================================
        WaterRAG Interactive App
========================================
This tool allows you to:
  1. Ask wastewater-treatment questions (RAG QA)
  2. Generate iterative literature reviews
  3. Inspect retrieved document sources
  4. Switch LLM models (gpt-4o, gpt-4.1)

Type 'help' for commands or 'exit' to quit.
========================================
"""


def print_help():
    print("""
Commands:
----------------------------------------
qa <question>
    → Answer question using RAG

review <topic>
    → Generate literature review

sources
    → Show last retrieved documents

setmodel <model>
    → Change model (gpt-4o, gpt-4.1)

help
    → Show this help

exit
    → Quit
""")


class WaterRAGApp:
    """Interactive WaterRAG application"""
    
    def __init__(self):
        print("Loading WaterRAG system...")
        
        # Initialize RAG QA module with LLM reranking
        self.rag = RAGSystem()
        
        # Initialize review module
        self.review_system = IterativeReviewSystem()
        
        self.last_docs = []
        self.current_model = self.rag.default_generation_model
        
        print(f"System loaded! Model: {self.current_model}\n")
    
    def qa(self, question):
        """Run RAG question answering with LLM reranking"""
        print("\n🔍 Running RAG QA with LLM reranking...\n")
        
        try:
            answer, docs = self.rag.query(
                question=question,
                top_k=10
            )
            self.last_docs = docs
            
            print("Answer:\n" + "="*60)
            print(answer)
            print(f"\nSources ({len(docs)}):")
            for i, doc in enumerate(docs, 1):
                citation = doc.metadata.get("citation_info", "No citation")
                score = doc.metadata.get("llm_rerank_score", 0)
                print(f"  [{i}] {citation} (score: {score:.2f})")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"Error: {e}\n")
    
    def run_review(self, topic):
        """Generate iterative literature review"""
        print("\n📘 Generating Review...\n")
        
        try:
            review, stats = self.review_system.generate_iterative_review(
                user_question=topic
            )
            
            # Get all documents used
            all_docs = stats.get('all_documents', [])
            self.last_docs = all_docs
            
            print("Review:\n" + "="*60)
            print(review)
            print("\nStats:")
            print(f"  Iterations: {stats['iterations']}")
            print(f"  Final score: {stats['final_score']}/10")
            print(f"  Documents used: {len(all_docs)}")
            
            # Display all sources
            if all_docs:
                print(f"\nSources ({len(all_docs)}):")
                for i, doc in enumerate(all_docs, 1):
                    citation = doc.metadata.get("citation_info", "No citation")
                    retrieval_type = doc.metadata.get("retrieval_iteration", "unknown")
                    print(f"  [{i}] {citation} ({retrieval_type})")
            
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"Error: {e}\n")
    
    def show_sources(self):
        """Display last retrieved sources"""
        if not self.last_docs:
            print("\nNo sources retrieved yet.\n")
            return
        
        print("\nLast Retrieved Sources:\n" + "="*60)
        for i, doc in enumerate(self.last_docs, 1):
            citation = doc.metadata.get("citation_info", "No citation")
            score = doc.metadata.get("llm_rerank_score") or doc.metadata.get("vector_score")
            if score is not None:
                print(f"{i}. {citation} [score: {score:.3f}]")
            else:
                print(f"{i}. {citation}")
        print("="*60 + "\n")
    
    def set_model(self, model_name):
        """Change LLM model"""
        print(f"\nSwitching to: {model_name}")
        
        try:
            # Update RAG system models
            self.rag.default_generation_model = model_name
            
            # Update review system model
            self.review_system.model = model_name
            self.review_system.retrieval_system.default_model = model_name
            
            self.current_model = model_name
            print("Model updated!\n")
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    print(WELCOME)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found in .env\n")
        return
    
    if not os.getenv("INDEX_PATH"):
        print("⚠️  INDEX_PATH not found in .env\n")
        return
    
    # Initialize app
    try:
        app = WaterRAGApp()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return
    
    # Main loop
    while True:
        try:
            user_input = input("WaterRAG> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Parse commands
        if user_input == "exit":
            print("Goodbye!")
            break
        
        elif user_input == "help":
            print_help()
        
        elif user_input.startswith("qa "):
            question = user_input[3:].strip()
            if question:
                app.qa(question)
            else:
                print("Usage: qa <question>\n")
        
        elif user_input.startswith("review "):
            topic = user_input[7:].strip()
            if topic:
                app.run_review(topic)
            else:
                print("Usage: review <topic>\n")
        
        elif user_input == "sources":
            app.show_sources()
        
        elif user_input.startswith("setmodel "):
            model = user_input[9:].strip()
            if model:
                app.set_model(model)
            else:
                print("Usage: setmodel <model_name>\n")
        
        else:
            print("Unknown command. Type 'help' for options.\n")


if __name__ == "__main__":
    main()