#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified RAG system with LLM reranking and quality score filtering
"""

import os
import json
import pickle
import logging
import requests
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class LLMReranker:
    """LLM document reranker with quality score filtering"""
    
    def __init__(
        self,
        api_key: str = None,
        default_model: str = None,
        batch_size: int = None,
        default_min_score: float = None
    ):
        """Initialize LLM reranker
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from .env)
            default_model: Default reranking model (defaults to DEFAULT_RERANK_MODEL from .env)
            batch_size: Batch size for processing (defaults to RERANKER_BATCH_SIZE from .env)
            default_min_score: Default minimum quality score threshold (defaults to DEFAULT_MIN_SCORE from .env)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.default_model = default_model or os.getenv("DEFAULT_RERANK_MODEL", "gpt-4.1-mini")
        self.batch_size = batch_size or int(os.getenv("RERANKER_BATCH_SIZE", "8"))
        self.default_min_score = default_min_score if default_min_score is not None else float(os.getenv("DEFAULT_MIN_SCORE", "0.5"))
        self.api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        logger.info(f"LLMReranker initialized - Model: {self.default_model}, Min score: {self.default_min_score}")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10,
        model: str = None,
        min_score: float = None
    ) -> List[Document]:
        """Rerank documents and filter by quality score
        
        Args:
            query: Query text
            documents: Candidate documents
            top_k: Maximum documents to return
            model: Reranking model
            min_score: Minimum score threshold
            
        Returns:
            Filtered and ranked documents (may be less than top_k)
        """
        if not documents:
            return []
        
        model_to_use = model or self.default_model
        min_score_to_use = min_score if min_score is not None else self.default_min_score
        
        logger.info(f"Reranking with model: {model_to_use}, min score: {min_score_to_use}")
        
        # Limit documents to process
        docs_to_process = documents[:min(len(documents), 100)]
        
        # Process in batches
        all_rankings = []
        for i in range(0, len(docs_to_process), self.batch_size):
            batch = docs_to_process[i:i + self.batch_size]
            try:
                rankings = self._rerank_batch(query, batch, i, model_to_use)
                all_rankings.extend(rankings)
            except Exception as e:
                logger.warning(f"Batch {i//self.batch_size + 1} reranking failed: {e}")
                # Use fallback rankings
                for j, doc in enumerate(batch):
                    all_rankings.append({
                        "doc_id": i + j + 1,
                        "new_rank": i + j + 1,
                        "relevance_score": 0.5,
                        "reason": f"Reranking failed: {str(e)}"
                    })
        
        # Apply rankings and filter
        return self._apply_rankings_with_filter(
            docs_to_process,
            all_rankings,
            top_k,
            min_score_to_use
        )
    
    def _rerank_batch(
        self,
        query: str,
        batch: List[Document],
        start_idx: int,
        rerank_model: str
    ) -> List[Dict]:
        """Rerank a batch of documents"""
        # Prepare document texts
        docs_text = ""
        for i, doc in enumerate(batch):
            doc_id = start_idx + i + 1
            content = doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content
            
            docs_text += f"""
Document {doc_id}:
- Source: {doc.metadata.get('document_id', 'unknown')}
- Citation: {doc.metadata.get('citation_info', 'No citation')}
- Content: {content}

"""
        
        system_prompt = """You are an expert document ranking system specializing in wastewater treatment research. Your task is to rerank documents based on their relevance to a query.

Instructions:
1. Analyze each document's relevance to the query
2. Consider topical relevance and information quality/specificity
3. Rank documents from most relevant (1) to least relevant
4. Provide a relevance score from 0.0 to 1.0 for each document
5. Give a brief reason for each document's ranking

SCORING GUIDELINES:
- 0.9-1.0: Directly answers query with specific, quantitative data
- 0.7-0.8: Highly relevant with useful specific information
- 0.5-0.6: Moderately relevant but may lack specificity
- 0.3-0.4: Tangentially related but not directly useful
- 0.0-0.2: Irrelevant or off-topic

Documents with scores below 0.5 may be filtered out, so be conservative with low scores.

Return JSON format:
{
    "rankings": [
        {
            "doc_id": 1,
            "new_rank": 1,
            "relevance_score": 0.95,
            "reason": "Directly addresses query with specific quantitative data..."
        },
        {
            "doc_id": 2,
            "new_rank": 2,
            "relevance_score": 0.82,
            "reason": "Relevant but less specific..."
        }
    ],
    "overall_analysis": "Brief summary of ranking decisions..."
}"""
        
        user_prompt = f"""Query: {query}

Documents to rank:
{docs_text}

Please rerank these documents based on their relevance. Focus on documents that provide specific, quantitative, or directly applicable information. Be careful with scoring - documents below 0.5 may be filtered out."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._call_api(messages, rerank_model)
        return self._parse_rankings(response)
    
    def _call_api(self, messages: List[Dict], api_model: str) -> str:
        """Call OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": api_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _parse_rankings(self, response_text: str) -> List[Dict]:
        """Parse LLM ranking response"""
        try:
            ranking_result = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            # Try extracting from markdown code block
            json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                ranking_result = json.loads(json_match.group(1))
            else:
                # Try finding any JSON object
                obj_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if obj_match:
                    ranking_result = json.loads(obj_match.group(0))
                else:
                    raise ValueError("Cannot parse JSON from LLM response")
        
        return ranking_result.get("rankings", [])
    
    def _apply_rankings_with_filter(
        self,
        documents: List[Document],
        rankings: List[Dict],
        top_k: int,
        min_score: float
    ) -> List[Document]:
        """Apply rankings and filter by score"""
        # Sort by new rank
        rankings.sort(key=lambda x: x.get("new_rank", 999))
        
        # Create document mapping
        doc_id_to_doc = {i + 1: doc for i, doc in enumerate(documents)}
        
        reranked_docs = []
        filtered_count = 0
        
        for ranking in rankings:
            doc_id = ranking["doc_id"]
            relevance_score = ranking.get("relevance_score", 0.5)
            
            # Apply score filter
            if relevance_score < min_score:
                filtered_count += 1
                continue
            
            # Check top_k limit
            if len(reranked_docs) >= top_k:
                break
            
            if doc_id in doc_id_to_doc:
                doc = doc_id_to_doc[doc_id]
                # Add reranking metadata
                doc.metadata["llm_rerank_score"] = relevance_score
                doc.metadata["llm_rerank_reason"] = ranking.get("reason", "No reason")
                doc.metadata["llm_new_rank"] = ranking.get("new_rank", 0)
                doc.metadata["ranking_method"] = "llm_reranker_filtered"
                doc.metadata["min_score_threshold"] = min_score
                reranked_docs.append(doc)
        
        final_count = len(reranked_docs)
        logger.info(f"Filtering complete - Filtered: {filtered_count}, Kept: {final_count}, Threshold: {min_score}")
        
        if final_count == 0:
            logger.warning(f"All documents filtered (threshold: {min_score}), consider lowering threshold")
        elif final_count < top_k:
            logger.info(f"Returned {final_count} docs (less than {top_k} due to filtering)")
        
        return reranked_docs


class RAGSystem:
    """RAG retrieval system with LLM reranking and score filtering"""
    
    def __init__(
        self,
        index_path: str = None,
        api_key: str = None,
        use_reranker: bool = True,
        default_rerank_model: str = None,
        default_generation_model: str = None,
        default_min_score: float = None
    ):
        """Initialize RAG system
        
        Args:
            index_path: Index directory path (defaults to INDEX_PATH from .env)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from .env)
            use_reranker: Enable LLM reranking
            default_rerank_model: Default reranking model (defaults to DEFAULT_RERANK_MODEL from .env)
            default_generation_model: Default generation model (defaults to DEFAULT_GENERATION_MODEL from .env)
            default_min_score: Default minimum score threshold (defaults to DEFAULT_MIN_SCORE from .env)
        """
        self.index_path = index_path or os.getenv("INDEX_PATH")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_reranker = use_reranker
        self.default_rerank_model = default_rerank_model or os.getenv("DEFAULT_RERANK_MODEL", "gpt-4.1-mini")
        self.default_generation_model = default_generation_model or os.getenv("DEFAULT_GENERATION_MODEL", "gpt-4.1")
        self.default_min_score = default_min_score if default_min_score is not None else float(os.getenv("DEFAULT_MIN_SCORE", "0.5"))
        
        # Initialize components
        self.faiss_index = self._load_faiss_index()
        self.bm25_retriever = self._load_bm25_retriever()
        self.reranker = LLMReranker(
            api_key=self.api_key,
            default_model=self.default_rerank_model,
            default_min_score=self.default_min_score
        ) if use_reranker else None
        
        logger.info(f"RAG system initialized - Reranker: {use_reranker}, Min score: {self.default_min_score}")
    
    def _load_faiss_index(self) -> Optional[FAISS]:
        """Load FAISS vector index"""
        try:
            faiss_path = os.path.join(self.index_path, "faiss_index")
            if not os.path.exists(faiss_path):
                logger.warning("FAISS index not found")
                return None
            
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                model_kwargs={'device': "cpu"},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"FAISS loading failed: {e}")
            return None
    
    def _load_bm25_retriever(self) -> Optional[BM25Retriever]:
        """Load BM25 retriever"""
        try:
            bm25_path = os.path.join(self.index_path, "bm25_retriever.pkl")
            if not os.path.exists(bm25_path):
                logger.warning("BM25 retriever not found")
                return None
            
            with open(bm25_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"BM25 loading failed: {e}")
            return None
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        initial_candidates: int = None,
        rerank_model: str = None,
        min_rerank_score: float = None
    ) -> List[Document]:
        """Retrieve relevant documents with score filtering
        
        Args:
            query: Query text
            top_k: Maximum documents to return
            initial_candidates: Initial retrieval candidates
            rerank_model: Reranking model
            min_rerank_score: Minimum reranking score threshold
            
        Returns:
            Filtered document list (may be less than top_k)
        """
        if initial_candidates is None:
            initial_candidates = top_k * 2
        
        if min_rerank_score is None:
            min_rerank_score = self.default_min_score
        
        candidates = []
        seen_ids = set()
        
        # Vector retrieval
        if self.faiss_index:
            try:
                vector_docs = self.faiss_index.similarity_search_with_score(query, k=initial_candidates)
                for doc, score in vector_docs:
                    doc_id = f"{doc.metadata.get('document_id', '')}-{doc.metadata.get('chunk_id', '')}"
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        doc.metadata["vector_score"] = float(score)
                        doc.metadata["retrieval_method"] = "vector"
                        candidates.append(doc)
            except Exception as e:
                logger.warning(f"Vector retrieval failed: {e}")
        
        # BM25 retrieval
        if self.bm25_retriever and len(candidates) < initial_candidates:
            try:
                bm25_docs = self.bm25_retriever.invoke(query)[:initial_candidates // 2]
                for doc in bm25_docs:
                    doc_id = f"{doc.metadata.get('document_id', '')}-{doc.metadata.get('chunk_id', '')}"
                    if doc_id not in seen_ids and len(candidates) < initial_candidates:
                        seen_ids.add(doc_id)
                        doc.metadata["bm25_retrieved"] = True
                        doc.metadata["retrieval_method"] = "bm25"
                        candidates.append(doc)
            except Exception as e:
                logger.warning(f"BM25 retrieval failed: {e}")
        
        # LLM reranking with score filtering
        if self.reranker and candidates:
            final_rerank_model = rerank_model or self.default_rerank_model
            candidates = self.reranker.rerank(
                query,
                candidates,
                top_k,
                final_rerank_model,
                min_rerank_score
            )
        else:
            candidates = candidates[:top_k]
        
        return candidates
    
    def generate_answer(
        self,
        query: str,
        documents: List[Document],
        generation_model: str = None
    ) -> str:
        """Generate answer from documents
        
        Args:
            query: Query text
            documents: Retrieved documents
            generation_model: Generation model
            
        Returns:
            Generated answer
        """
        if not documents:
            return "No relevant documents found. Possible reasons: 1) No relevant content retrieved, or 2) All documents filtered due to low quality scores. Try rephrasing or lowering the threshold."
        
        model_to_use = generation_model or self.default_generation_model
        
        # Group documents by citation to avoid duplication
        citation_groups: Dict[str, Dict[str, Any]] = {}
        for doc in documents:
            raw_citation = doc.metadata.get('citation_info', 'Unknown Source')
            
            # Normalize to hashable key
            if isinstance(raw_citation, (dict, list)):
                citation_key = json.dumps(raw_citation, ensure_ascii=False, sort_keys=True)
            else:
                citation_key = str(raw_citation)
            
            if citation_key not in citation_groups:
                citation_groups[citation_key] = {
                    'contents': [],
                    'max_llm_score': doc.metadata.get('llm_rerank_score', 0),
                    'min_score_threshold': doc.metadata.get('min_score_threshold', 'N/A'),
                    'raw_citation': raw_citation,
                }
            
            citation_groups[citation_key]['contents'].append(doc.page_content)
            current_score = doc.metadata.get('llm_rerank_score', 0)
            if current_score > citation_groups[citation_key]['max_llm_score']:
                citation_groups[citation_key]['max_llm_score'] = current_score
        
        # Sort by LLM score
        sorted_citations = sorted(
            citation_groups.items(),
            key=lambda x: x[1]['max_llm_score'],
            reverse=True
        )
        
        # Build context with quality scores
        doc_contexts = []
        for citation_key, group_data in sorted_citations:
            label = group_data.get('raw_citation', 'Unknown Source')
            if isinstance(label, (dict, list)):
                label = json.dumps(label, ensure_ascii=False)
            
            merged_content = "\n\n".join(group_data['contents'])
            doc_context = f"""Citation: {label}
Quality Score: {group_data['max_llm_score']:.2f}
Content: {merged_content}

"""
            doc_contexts.append(doc_context)
        
        full_context = "\n".join(doc_contexts)
        
        system_prompt = (
            "You are an expert in wastewater treatment. Answer based on provided context.\n"
            "Documents are intelligently reranked by relevance, filtered by quality score, and grouped by source.\n"
            "Each source includes a Quality Score (0.0-1.0) indicating relevance.\n\n"
            "Requirements:\n"
            "1. If context is insufficient, state \"Based on provided information, I cannot answer this question\".\n"
            "2. Synthesize insights across multiple sources for coherent, comprehensive answers.\n"
            "3. Prioritize quantitative data, metrics, or experimental results whenever possible.\n"
            "4. Answer based on context only. Do not fabricate information.\n"
        )
        
        user_prompt = f"""Question: {query}

Context Information (Sources ranked by relevance, filtered by quality, and grouped):
{full_context}

Please provide a comprehensive answer based only on the provided context.
Reference sources directly using their citations when making specific claims.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model_to_use,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 4096
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}"
    
    def query(
        self,
        question: str,
        top_k: int = 10,
        initial_candidates: int = None,
        min_rerank_score: float = None,
        rerank_model: str = None,
        generation_model: str = None
    ) -> Tuple[str, List[Document]]:
        """Complete query workflow: retrieve → rerank → filter → generate
        
        Args:
            question: Query question
            top_k: Maximum final documents
            initial_candidates: Initial retrieval candidates (default: top_k*2)
            min_rerank_score: Minimum reranking score threshold (default: 0.5)
            rerank_model: Reranking model (e.g., "gpt-4.1-mini", "gpt-4.1")
            generation_model: Answer generation model (e.g., "gpt-4.1", "gpt-4.1-mini")
            
        Returns:
            (answer, documents): Answer and quality-filtered documents
        """
        # Retrieve documents with score filtering
        documents = self.retrieve(
            question,
            top_k,
            initial_candidates,
            rerank_model,
            min_rerank_score
        )
        
        # Generate answer
        answer = self.generate_answer(question, documents, generation_model)
        
        # Log parameters used
        used_rerank_model = rerank_model or self.default_rerank_model
        used_generation_model = generation_model or self.default_generation_model
        used_initial_candidates = initial_candidates or top_k * 2
        used_min_score = min_rerank_score if min_rerank_score is not None else self.default_min_score
        
        # Log quality statistics
        if documents:
            avg_score = sum(doc.metadata.get('llm_rerank_score', 0) for doc in documents) / len(documents)
            min_doc_score = min(doc.metadata.get('llm_rerank_score', 0) for doc in documents)
            max_doc_score = max(doc.metadata.get('llm_rerank_score', 0) for doc in documents)
            
            logger.info(f"Query complete - Candidates: {used_initial_candidates}, Final docs: {len(documents)}")
            logger.info(f"Quality stats - Threshold: {used_min_score}, Avg: {avg_score:.3f}, Range: [{min_doc_score:.3f}-{max_doc_score:.3f}]")
        else:
            logger.warning(f"Query complete but no documents - Threshold: {used_min_score} (may be too strict)")
        
        return answer, documents


if __name__ == "__main__":
    # Usage example - loads from .env by default
    rag = RAGSystem()
    
    # Basic query with default threshold (0.5)
    answer, docs = rag.query(
        "What are the main factors affecting nitrogen removal?",
        top_k=5
    )
    
    print(f"Answer: {answer}")
    print(f"Used {len(docs)} high-quality documents")
    
    # Strict filtering (threshold 0.7)
    strict_answer, strict_docs = rag.query(
        "What are optimal denitrification conditions?",
        top_k=8,
        min_rerank_score=0.7,
        rerank_model="gpt-4.1"
    )
    
    print(f"\nStrict filtering: {len(strict_docs)} docs with threshold 0.7")