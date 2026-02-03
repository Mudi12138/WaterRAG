import os
import json
import time
import logging
import pickle
import torch
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RetrievalSystem")


class RetrievalSystem:
    """Simplified retrieval system with core functionality
    
    Features:
    - Vector search (FAISS)
    - BM25 search
    - Hybrid retrieval
    - GPT-4o and GPT-4.1 support
    """
    
    def __init__(
        self,
        index_path: str = None,
        openai_api_key: str = None,
        openai_api_url: str = None,
        default_model: str = None,
        default_chunk_top_k: int = None,
        default_final_top_k: int = None,
        max_tokens: int = None
    ):
        """Initialize retrieval system
        
        Args:
            index_path: Path to indexed files (defaults to INDEX_PATH from .env)
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY from .env)
            openai_api_url: OpenAI API endpoint (defaults to OPENAI_API_URL from .env)
            default_model: Default model (defaults to DEFAULT_GENERATION_MODEL from .env)
            default_chunk_top_k: Initial retrieval candidates (defaults to DEFAULT_CHUNK_TOP_K from .env)
            default_final_top_k: Final results after merging (defaults to DEFAULT_FINAL_TOP_K from .env)
            max_tokens: Maximum tokens for generation (defaults to DEFAULT_MAX_TOKENS from .env)
        """
        self.index_path = index_path or os.getenv("INDEX_PATH")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_api_url = openai_api_url or os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        self.default_model = default_model or os.getenv("DEFAULT_GENERATION_MODEL", "gpt-4o")
        self.max_tokens = max_tokens or int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))
        
        # Retrieval parameters
        self.default_chunk_top_k = default_chunk_top_k or int(os.getenv("DEFAULT_CHUNK_TOP_K", "20"))
        self.default_final_top_k = default_final_top_k or int(os.getenv("DEFAULT_FINAL_TOP_K", "10"))
        
        # Supported models
        self.supported_models = {
            "gpt-4o": {
                "max_tokens": 4096,
                "context_window": 128000,
            },
            "gpt-4.1": {
                "max_tokens": 32000,
                "context_window": 128000,
            }
        }
        
        # Load indexes
        self.faiss_index = None
        self.bm25_retriever = None
        self._load_indexes()
        
        logger.info(f"System initialized - Model: {self.default_model}")
    
    def _load_indexes(self):
        """Load FAISS and BM25 indexes"""
        try:
            # Load FAISS index
            faiss_path = os.path.join(self.index_path, "faiss_index")
            if os.path.exists(faiss_path):
                model_kwargs = {'device': "cuda" if torch.cuda.is_available() else "cpu"}
                encode_kwargs = {'normalize_embeddings': True}
                
                embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-large-en-v1.5",
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                
                self.faiss_index = FAISS.load_local(
                    faiss_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS index loaded")
            
            # Load BM25 retriever
            bm25_path = os.path.join(self.index_path, "bm25_retriever.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, 'rb') as f:
                    self.bm25_retriever = pickle.load(f)
                logger.info("BM25 retriever loaded")
        
        except Exception as e:
            logger.error(f"Error loading indexes: {e}")
            raise
    
    def vector_search(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """Vector search using FAISS
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        if not self.faiss_index:
            return []
        
        try:
            docs_with_scores = self.faiss_index.similarity_search_with_score(query, k=top_k)
            return docs_with_scores
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def bm25_search(self, query: str, top_k: int = 20) -> List[Document]:
        """BM25 search
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of documents
        """
        if not self.bm25_retriever:
            return []
        
        try:
            docs = self.bm25_retriever.invoke(query)[:top_k]
            return docs
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []
    
    def retrieve(
        self,
        query: str,
        chunk_top_k: int = None,
        final_top_k: int = None
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Retrieve documents using hybrid search
        
        Args:
            query: Query text
            chunk_top_k: Initial candidates
            final_top_k: Final results
            
        Returns:
            (documents, statistics)
        """
        chunk_top_k = chunk_top_k or self.default_chunk_top_k
        final_top_k = final_top_k or self.default_final_top_k
        
        start_time = time.time()
        stats = {"query": query, "success": False}
        
        try:
            # Vector search
            vector_docs = self.vector_search(query, top_k=chunk_top_k)
            
            # BM25 search
            bm25_docs = self.bm25_search(query, top_k=chunk_top_k)
            
            # Merge results (deduplicate)
            seen_ids = set()
            candidates = []
            
            # Add vector results
            for doc, score in vector_docs:
                doc_id = f"{doc.metadata.get('document_id', '')}-{doc.metadata.get('chunk_id', '')}"
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    doc.metadata["vector_score"] = float(score)
                    candidates.append(doc)
            
            # Add BM25 results
            for doc in bm25_docs:
                doc_id = f"{doc.metadata.get('document_id', '')}-{doc.metadata.get('chunk_id', '')}"
                if len(candidates) < chunk_top_k and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    doc.metadata["bm25_retrieved"] = True
                    candidates.append(doc)
            
            # Sort by vector score and take top_k
            candidates.sort(key=lambda d: d.metadata.get("vector_score", 999.0))
            results = candidates[:final_top_k]
            
            stats.update({
                "success": True,
                "total_time": time.time() - start_time,
                "vector_results": len(vector_docs),
                "bm25_results": len(bm25_docs),
                "final_results": len(results)
            })
            
            return results, stats
        
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            stats["error"] = str(e)
            return [], stats
    
    def call_openai_api(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.2,
        max_tokens: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Call OpenAI API
        
        Args:
            messages: Message list
            model: Model name (gpt-4o or gpt-4.1)
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            **kwargs: Additional API parameters
            
        Returns:
            API response
        """
        import requests
        
        model_to_use = model or self.default_model
        max_tokens_to_use = max_tokens or self.max_tokens
        
        # Validate model
        if model_to_use not in self.supported_models:
            raise ValueError(f"Unsupported model: {model_to_use}")
        
        # Check token limit
        model_config = self.supported_models[model_to_use]
        if max_tokens_to_use > model_config["max_tokens"]:
            max_tokens_to_use = model_config["max_tokens"]
        
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens_to_use
        }
        
        # Add additional parameters
        for key in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop']:
            if key in kwargs:
                data[key] = kwargs[key]
        
        try:
            response = requests.post(self.openai_api_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"API call error: {e}")
            raise
    
    def generate_answer(
        self,
        query: str,
        documents: List[Document],
        model: str = None,
        system_prompt: str = None,
        temperature: float = 0.2,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """Generate answer using retrieved documents
        
        Args:
            query: User query
            documents: Retrieved documents
            model: Model name
            system_prompt: Custom system prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            Generated answer
        """
        if not documents:
            return "No relevant documents found."
        
        if not self.openai_api_key:
            return "OpenAI API key not provided."
        
        model_to_use = model or self.default_model
        
        # Prepare document context
        doc_contexts = []
        for i, doc in enumerate(documents, 1):
            doc_id = doc.metadata.get('document_id', 'unknown')
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            citation = doc.metadata.get('citation_info', 'No citation')
            
            doc_context = (
                f"[Document {i}]\n"
                f"Source: {doc_id}\n"
                f"Chunk: {chunk_id}\n"
                f"Citation: {citation}\n"
                f"Content:\n{doc.page_content}\n"
            )
            doc_contexts.append(doc_context)
        
        full_context = "\n".join(doc_contexts)
        
        # Default system prompt
        if not system_prompt:
            system_prompt = (
                "You are an expert in wastewater treatment. Answer based on provided context.\n"
                "Requirements:\n"
                "1. State 'Cannot answer based on provided information' if context is insufficient.\n"
                "2. Synthesize insights across sources for comprehensive answers.\n"
                "3. Prioritize quantitative data and experimental results.\n"
                "4. Only use provided context - do not fabricate information.\n"
                "5. Structure answer clearly with proper citations."
            )
        
        user_prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{full_context}\n\n"
            f"Provide a comprehensive answer based only on the context."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.call_openai_api(
                messages,
                model=model_to_use,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return f"Error generating answer: {str(e)}"
    
    def retrieve_and_answer(
        self,
        query: str,
        chunk_top_k: int = None,
        final_top_k: int = None,
        model: str = None,
        **kwargs
    ) -> Tuple[str, List[Document]]:
        """Retrieve documents and generate answer
        
        Args:
            query: User query
            chunk_top_k: Initial candidates
            final_top_k: Final results
            model: Model name
            **kwargs: Additional parameters
            
        Returns:
            (answer, retrieved_documents)
        """
        # Retrieve documents
        results, stats = self.retrieve(
            query,
            chunk_top_k=chunk_top_k,
            final_top_k=final_top_k
        )
        
        if not results:
            return "No relevant documents found.", []
        
        # Generate answer
        answer = self.generate_answer(
            query,
            results,
            model=model,
            **kwargs
        )
        
        return answer, results


def create_system(
    index_path: str = None,
    openai_api_key: str = None,
    default_model: str = None,
    **kwargs
) -> RetrievalSystem:
    """Create retrieval system instance
    
    Args:
        index_path: Index path (defaults to .env)
        openai_api_key: OpenAI API key (defaults to .env)
        default_model: Default model (defaults to .env)
        **kwargs: Additional parameters
        
    Returns:
        RetrievalSystem instance
    """
    system = RetrievalSystem(
        index_path=index_path,
        openai_api_key=openai_api_key,
        default_model=default_model,
        **kwargs
    )
    
    return system


if __name__ == "__main__":
    # Usage example - loads from .env by default
    print("=== Simplified Retrieval System ===")
    print("Supported models: gpt-4o, gpt-4.1")
    print("\nUsage:")
    print("""
# Create system (loads from .env)
system = create_system()

# Or override specific parameters
system = create_system(
    default_model="gpt-4.1"
)

# Retrieve and answer
answer, docs = system.retrieve_and_answer(
    query="Your question here",
    model="gpt-4.1"
)
    """)