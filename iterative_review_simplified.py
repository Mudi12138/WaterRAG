#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified iterative review generation system:
1. Query expansion → retrieval → filter → initial review
2. Review audit agent evaluates quality
3. If insufficient → supplement retrieval → update review
4. Iterate until adequate or max iterations reached
"""

import json
import os
import time
import logging
import requests
from typing import List, Dict, Any, Tuple, Set
from dotenv import load_dotenv
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Import simplified retrieval system
from retrieval_simplified import RetrievalSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("IterativeReview")


class IterativeReviewSystem:
    """Iterative review generation with adaptive supplementation
    
    Flow:
    1. Query expansion → initial retrieval → filter → generate review
    2. Audit agent evaluates review quality
    3. If insufficient → identify missing aspects → supplement retrieval → update
    4. Repeat until adequate or max iterations
    """
    
    def __init__(
        self,
        index_path: str = None,
        openai_api_key: str = None,
        openai_api_url: str = None,
        model: str = None
    ):
        """Initialize iterative review system
        
        Args:
            index_path: Index directory path (defaults to INDEX_PATH from .env)
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY from .env)
            openai_api_url: API endpoint URL (defaults to OPENAI_API_URL from .env)
            model: Model to use (defaults to DEFAULT_GENERATION_MODEL from .env)
        """
        self.index_path = index_path or os.getenv("INDEX_PATH")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_api_url = openai_api_url or os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        self.model = model or os.getenv("DEFAULT_GENERATION_MODEL", "gpt-4.1")
        
        # Initialize retrieval system
        self.retrieval_system = RetrievalSystem(
            index_path=self.index_path,
            openai_api_key=self.openai_api_key,
            openai_api_url=self.openai_api_url,
            default_model=self.model
        )
        
        # Track retrieved documents to avoid duplicates
        self.retrieved_document_ids = set()
        
        logger.info(f"System initialized - Model: {self.model}")
    
    def call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """Call OpenAI API
        
        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            
        Returns:
            API response
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(self.openai_api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API call error: {e}")
            raise
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from API response"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            # Try extracting from markdown code block
            json_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            # Try finding any JSON object
            obj_match = re.search(r"\{.*\}", content, re.DOTALL)
            if obj_match:
                return json.loads(obj_match.group(0))
            logger.warning("JSON parsing failed")
            return {}
    
    # ==================== Query Expansion ====================
    def expand_query(self, user_question: str) -> Tuple[str, Dict[str, Any]]:
        """Expand query for better retrieval
        
        Args:
            user_question: Original user query
            
        Returns:
            (expanded_query, expansion_stats)
        """
        system_prompt = (
            "You are a search query optimization expert for scientific literature retrieval. "
            "Expand queries to improve retrieval while preserving original meaning."
        )
        
        user_prompt = f"""
Given the following research question, please expand and optimize it for better literature retrieval effectiveness.

Original Query: "{user_question}"

Your task is to:
1. **Preserve the original meaning** - do not change the core intent or scope
2. **Add relevant technical terms** - include appropriate scientific terminology
3. **Make it more specific** - add context that helps target relevant literature
4. **Improve keyword coverage** - ensure important concepts are well-represented
5. **Keep it focused** - avoid over-expansion that might dilute the search

Guidelines:
- Add synonyms and related technical terms
- Include relevant process names, methods, or technologies
- Add context about the application domain (wastewater treatment)
- Ensure the expanded query is still coherent and readable
- Aim for 2-3x the length of the original query but no more

Return your response in JSON format:
{{
  "expanded_query": "the optimized search query for retrieval",
  "added_terms": ["list", "of", "key", "terms", "added"],
  "expansion_rationale": "brief explanation of why these terms help retrieval"
}}

Please expand and optimize this query for literature retrieval:
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.call_api(messages, max_tokens=1500)
            content = response["choices"][0]["message"]["content"]
            result = self._parse_json_response(content)
            
            if not result or "expanded_query" not in result:
                return user_question, {"success": False}
            
            expanded = result.get("expanded_query", user_question)
            stats = {
                "success": True,
                "original": user_question,
                "expanded": expanded,
                "added_terms": result.get("added_terms", []),
                "ratio": len(expanded.split()) / max(1, len(user_question.split()))
            }
            
            logger.info(f"Query expanded: {len(user_question.split())} → {len(expanded.split())} words")
            return expanded, stats
        
        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return user_question, {"success": False, "error": str(e)}
    
    # ==================== Initial Retrieval ====================
    def initial_retrieval_and_review(
        self,
        user_question: str,
        initial_candidates: int = 30,
        initial_top_k: int = 20,
        enable_expansion: bool = True
    ) -> Tuple[str, List[Document], Dict[str, Any]]:
        """Initial retrieval and review generation
        
        Args:
            user_question: User query
            initial_candidates: Initial retrieval candidates
            initial_top_k: Documents to keep after retrieval
            enable_expansion: Enable query expansion
            
        Returns:
            (review_text, documents, statistics)
        """
        start_time = time.time()
        self.retrieved_document_ids = set()
        
        # Query expansion
        if enable_expansion:
            expanded_query, expansion_stats = self.expand_query(user_question)
            retrieval_query = expanded_query
        else:
            retrieval_query = user_question
            expansion_stats = {"success": False}
        
        # Retrieve documents
        documents, retrieval_stats = self.retrieval_system.retrieve(
            query=retrieval_query,
            chunk_top_k=initial_candidates,
            final_top_k=initial_top_k
        )
        
        # Track retrieved documents
        for doc in documents:
            doc_id = f"{doc.metadata.get('document_id', 'unknown')}_{doc.metadata.get('chunk_id', 'unknown')}"
            self.retrieved_document_ids.add(doc_id)
            doc.metadata["retrieval_iteration"] = "initial"
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        # Filter documents
        filtered_documents = self.filter_documents(documents, user_question)
        logger.info(f"Filtered to {len(filtered_documents)} documents")
        
        # Generate initial review
        initial_review = self.generate_review(
            user_question,
            filtered_documents,
            review_type="initial"
        )
        
        stats = {
            "processing_time": time.time() - start_time,
            "documents_retrieved": len(documents),
            "documents_filtered": len(filtered_documents),
            "expansion_stats": expansion_stats
        }
        
        return initial_review, filtered_documents, stats
    
    def filter_documents(
        self,
        documents: List[Document],
        user_question: str
    ) -> List[Document]:
        """Filter documents using GPT-4.1
        
        Args:
            documents: Documents to filter
            user_question: Original query
            
        Returns:
            Filtered documents
        """
        if not documents:
            return []
        
        logger.info(f"Filtering {len(documents)} documents")
        
        # Process in batches
        filtered_docs = []
        batch_size = 5
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Prepare document texts
            doc_texts = []
            for j, doc in enumerate(batch):
                doc_id = doc.metadata.get('document_id', 'unknown')
                doc_texts.append(f"Document {j+1} (ID: {doc_id}):\n{doc.page_content}")
            
            doc_content = "\n\n" + "="*50 + "\n\n".join(doc_texts)
            
            system_prompt = (
                "You are a document relevance filter. Determine if each document contains "
                "useful information relevant to the research question."
            )
            
            user_prompt = f"""Research Question: "{user_question}"

Please evaluate each document to determine if it contains useful information relevant to this research question.

For each document, determine:
1. Is the content directly relevant to the research question?
2. Does it contain substantial information (data, methods, findings, insights)?
3. Would it meaningfully contribute to a literature review on this topic?

Return your evaluation in JSON format:
{{
  "Document 1": {{
    "useful": true/false,
    "relevance_score": 0-10,
    "reason": "brief explanation of why it's useful/not useful"
  }},
  "Document 2": {{
    "useful": true/false,
    "relevance_score": 0-10,
    "reason": "brief explanation"
  }}
}}

Only mark as "useful: true" if the document:
- Contains substantial information directly related to the research question
- Provides valuable data, methods, findings, or insights
- Would meaningfully contribute to a literature review

Documents to evaluate:
{doc_content}
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            try:
                response = self.call_api(messages, temperature=0.1, max_tokens=1500)
                content = response["choices"][0]["message"]["content"]
                evaluation = self._parse_json_response(content)
                
                # Filter based on evaluation
                for j, doc in enumerate(batch):
                    doc_key = f"Document {j+1}"
                    doc_eval = evaluation.get(doc_key, {"useful": True, "relevance_score": 5})
                    
                    if doc_eval.get("useful", True):
                        doc.metadata["filter_useful"] = True
                        doc.metadata["relevance_score"] = doc_eval.get("relevance_score", 5)
                        filtered_docs.append(doc)
            
            except Exception as e:
                logger.error(f"Filtering batch error: {e}")
                # Keep documents on error
                for doc in batch:
                    doc.metadata["filter_useful"] = True
                    doc.metadata["relevance_score"] = 5
                filtered_docs.extend(batch)
        
        return filtered_docs
    
    def generate_review(
        self,
        user_question: str,
        documents: List[Document],
        review_type: str = "initial"
    ) -> str:
        """Generate review from documents
        
        Args:
            user_question: User query
            documents: Document list
            review_type: "initial" or "updated"
            
        Returns:
            Generated review text
        """
        if not documents:
            return "No relevant documents found."
        
        logger.info(f"Generating {review_type} review from {len(documents)} documents")
        
        # Prepare document context
        doc_contexts = []
        for i, doc in enumerate(documents, 1):
            doc_id = doc.metadata.get('document_id', 'unknown')
            citation = doc.metadata.get('citation_info', 'No citation')
            doc_contexts.append(
                f"[Document {i}]\nSource: {doc_id}\nCitation: {citation}\nContent:\n{doc.page_content}\n"
            )
        
        full_context = "\n".join(doc_contexts)
        
        system_prompt = (
            "You are an expert scientific reviewer in wastewater treatment. Write comprehensive "
            "literature reviews that directly address research questions."
        )
        
        if review_type == "initial":
            instruction = (
                "Write a comprehensive literature review with critical evaluation, comparison, "
                "discussion of findings, and identification of research gaps."
            )
        else:
            instruction = (
                "This is an updated review. Integrate new information seamlessly while "
                "maintaining coherence and flow."
            )
        
        user_prompt = f"""
Research Question: "{user_question}"

{instruction}

Your review should include:
- Critical evaluation of the current state of knowledge
- Comparison of different approaches, methodologies, and key findings
- Discussion of technical mechanisms, quantitative data, and practical implications
- Identification of research gaps and future directions
- Clear synthesis of insights that advance understanding

Write your review as a flowing, coherent discussion that naturally incorporates insights from the literature. Focus on creating logical connections between ideas and studies. Ensure your analysis is substantive, technically sound, and provides meaningful insights.

**Literature Context:**
{full_context}

Please provide a comprehensive literature review that thoroughly addresses: {user_question}
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.call_api(messages, max_tokens=6000)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Review generation error: {e}")
            return f"Error generating review: {str(e)}"
    
    # ==================== Review Audit ====================
    def audit_review(
        self,
        user_question: str,
        current_review: str
    ) -> Tuple[bool, int, List[str], str]:
        """Audit review quality and identify gaps
        
        Args:
            user_question: Original query
            current_review: Current review text
            
        Returns:
            (adequate, score, missing_aspects, reason)
        """
        system_prompt = (
            "You are an expert review auditor. Evaluate if literature reviews adequately "
            "address research questions. Apply strict standards."
        )
        
        user_prompt = f"""
Please audit the following literature review to determine if it adequately addresses the research question. **Apply STRICT evaluation standards - be critical and demanding.**

**Research Question**: "{user_question}"

**Current Literature Review**:
{current_review}

**STRICT Evaluation Criteria** (All must be met for high scores):
1. **Completeness** (0-2 points): Does the review comprehensively cover ALL critical aspects? Missing even one important dimension significantly reduces the score.
2. **Depth** (0-2 points): Is each aspect discussed with substantial technical depth, specific parameters, quantitative data, and detailed mechanisms? Surface-level discussion gets low scores.
3. **Evidence** (0-2 points): Are claims supported by multiple, recent, high-quality sources? Each major point needs robust literature backing.
4. **Critical Analysis** (0-2 points): Does the review provide thorough comparison, identify contradictions, discuss limitations, and synthesize insights? Mere description without analysis gets low scores.
5. **Technical Rigor** (0-2 points): Are specific methods, parameters, performance data, and technical details included? Vague generalizations are penalized.

**STRICT Scoring Guidelines**:
- **9-10**: Exceptional quality, publication-ready, addresses ALL aspects comprehensively with deep technical analysis
- **7-8**: Good quality but has notable gaps or lacks depth in some areas
- **5-6**: Adequate but missing important aspects or lacks sufficient technical detail
- **3-4**: Basic coverage but significant deficiencies in depth, evidence, or completeness
- **1-2**: Poor quality, major gaps, insufficient evidence or analysis

**Your Tasks** (Be demanding and critical):
1. Rate the review's adequacy on a scale of 1-10 using the STRICT criteria above
2. Determine if the review is sufficient (score ≥ 7) or needs improvement
3. If improvement needed, specify the exact missing aspects that should be supplemented

**Return Format** (JSON):
{{
  "adequate": true/false,
  "score": 1-10,
  "missing_aspects": [
    "specific aspect 1 that needs to be added",
    "specific aspect 2 that needs to be added"
  ],
  "audit_reason": "detailed explanation of the evaluation and what's missing"
}}

**Guidelines for Missing Aspects**:
- Be specific and actionable (e.g., "quantitative performance comparison of different nitrogen removal technologies with specific removal efficiency data" rather than "more performance data")
- Focus on substantial gaps that significantly impact review quality
- Consider technical, environmental, economic, operational, regulatory, and future research angles
- Each aspect should be distinct, searchable, and address a meaningful gap
- Prioritize aspects that would elevate the review from good to excellent

**IMPORTANT**: Be critical and demanding. A score of 8+ should only be given to truly comprehensive, technically rigorous reviews. Most reviews will have room for improvement and should score 6-7. Don't hesitate to give lower scores for reviews with significant gaps or insufficient depth.

Please provide your audit evaluation:
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.call_api(messages, max_tokens=2000)
            content = response["choices"][0]["message"]["content"]
            result = self._parse_json_response(content)
            
            if not result:
                return False, 5, ["additional analysis"], "Audit parsing failed"
            
            adequate = result.get("adequate", False)
            score = result.get("score", 5)
            missing_aspects = result.get("missing_aspects", [])
            reason = result.get("audit_reason", "No reason provided")
            
            logger.info(f"Audit: adequate={adequate}, score={score}/10")
            return adequate, score, missing_aspects, reason
        
        except Exception as e:
            logger.error(f"Audit error: {e}")
            return False, 5, ["comprehensive analysis"], f"Error: {str(e)}"
    
    # ==================== Supplementary Retrieval ====================
    def supplementary_retrieval(
        self,
        missing_aspects: List[str],
        candidates_per_aspect: int = 15,
        top_k_per_aspect: int = 10
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Retrieve documents for missing aspects
        
        Args:
            missing_aspects: List of missing aspects
            candidates_per_aspect: Candidates per aspect
            top_k_per_aspect: Documents to keep per aspect
            
        Returns:
            (new_documents, statistics)
        """
        logger.info(f"Supplementary retrieval for {len(missing_aspects)} aspects")
        
        start_time = time.time()
        all_new_documents = []
        
        for i, aspect in enumerate(missing_aspects):
            logger.info(f"Retrieving aspect [{i+1}/{len(missing_aspects)}]: {aspect}")
            
            try:
                documents, stats = self.retrieval_system.retrieve(
                    query=aspect,
                    chunk_top_k=candidates_per_aspect,
                    final_top_k=top_k_per_aspect
                )
                
                # Filter duplicates
                new_documents = []
                for doc in documents:
                    doc_id = f"{doc.metadata.get('document_id', 'unknown')}_{doc.metadata.get('chunk_id', 'unknown')}"
                    
                    if doc_id not in self.retrieved_document_ids:
                        doc.metadata["supplementary_aspect"] = aspect
                        doc.metadata["retrieval_iteration"] = "supplementary"
                        new_documents.append(doc)
                        self.retrieved_document_ids.add(doc_id)
                
                all_new_documents.extend(new_documents)
                logger.info(f"Added {len(new_documents)} new documents for aspect")
            
            except Exception as e:
                logger.error(f"Retrieval error for aspect '{aspect}': {e}")
        
        # Filter new documents
        if all_new_documents:
            combined_query = f"Research aspects: {'; '.join(missing_aspects)}"
            filtered_new_documents = self.filter_documents(all_new_documents, combined_query)
            logger.info(f"Filtered: {len(all_new_documents)} → {len(filtered_new_documents)} documents")
        else:
            filtered_new_documents = []
        
        stats = {
            "processing_time": time.time() - start_time,
            "total_new": len(all_new_documents),
            "filtered": len(filtered_new_documents)
        }
        
        return filtered_new_documents, stats
    
    def update_review(
        self,
        user_question: str,
        current_review: str,
        new_documents: List[Document],
        missing_aspects: List[str]
    ) -> str:
        """Update review with new documents
        
        Args:
            user_question: Original query
            current_review: Current review
            new_documents: New documents
            missing_aspects: Missing aspects
            
        Returns:
            Updated review
        """
        if not new_documents:
            return current_review
        
        logger.info(f"Updating review with {len(new_documents)} new documents")
        
        # Prepare new document context
        new_doc_contexts = []
        for i, doc in enumerate(new_documents, 1):
            doc_id = doc.metadata.get('document_id', 'unknown')
            citation = doc.metadata.get('citation_info', 'No citation')
            aspect = doc.metadata.get('supplementary_aspect', 'unknown')
            new_doc_contexts.append(
                f"[New Document {i} - for {aspect}]\nSource: {doc_id}\nCitation: {citation}\nContent:\n{doc.page_content}\n"
            )
        
        new_docs_content = "\n".join(new_doc_contexts)
        
        system_prompt = (
            "You are an expert scientific reviewer. Update existing reviews by integrating "
            "new information while maintaining coherence."
        )
        
        user_prompt = f"""
**Research Question**: "{user_question}"

**Current Literature Review**:
{current_review}

**Missing Aspects Identified**: {', '.join(missing_aspects)}

**New Literature to Integrate**:
{new_docs_content}

**Task**: Update the literature review by seamlessly integrating the new information. Your updated review should:

1. **Address the missing aspects** identified in the audit
2. **Maintain coherence** and flow with the existing content
3. **Integrate new information naturally** without creating obvious seams
4. **Enhance overall quality** by filling gaps and providing more comprehensive coverage
5. **Preserve valuable content** from the original review
6. **Ensure technical accuracy** and appropriate scientific tone

**Instructions**:
- Do not simply append new sections; integrate information throughout where it fits best
- Maintain logical structure and smooth transitions
- Ensure all aspects of the research question are now well-addressed
- Keep the review comprehensive but focused

Please provide the updated literature review:
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.call_api(messages, max_tokens=8000)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Update error: {e}")
            return current_review
    
    # ==================== Main Iterative Method ====================
    def generate_iterative_review(
        self,
        user_question: str,
        max_iterations: int = None,
        min_score_threshold: int = None,
        initial_candidates: int = None,
        initial_top_k: int = None,
        supplement_candidates: int = None,
        supplement_top_k: int = None,
        enable_expansion: bool = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate iterative review with adaptive supplementation
        
        Args:
            user_question: User query
            max_iterations: Maximum iterations (defaults to REVIEW_MAX_ITERATIONS from .env)
            min_score_threshold: Minimum acceptable score (defaults to REVIEW_MIN_SCORE_THRESHOLD from .env)
            initial_candidates: Initial retrieval candidates (defaults to REVIEW_INITIAL_CANDIDATES from .env)
            initial_top_k: Initial documents to keep (defaults to REVIEW_INITIAL_TOP_K from .env)
            supplement_candidates: Supplement candidates per aspect (defaults to REVIEW_SUPPLEMENT_CANDIDATES from .env)
            supplement_top_k: Supplement documents per aspect (defaults to REVIEW_SUPPLEMENT_TOP_K from .env)
            enable_expansion: Enable query expansion (defaults to REVIEW_ENABLE_EXPANSION from .env)
            
        Returns:
            (final_review, statistics)
        """
        # Load from .env with defaults
        max_iterations = max_iterations if max_iterations is not None else int(os.getenv("REVIEW_MAX_ITERATIONS", "3"))
        min_score_threshold = min_score_threshold if min_score_threshold is not None else int(os.getenv("REVIEW_MIN_SCORE_THRESHOLD", "7"))
        initial_candidates = initial_candidates if initial_candidates is not None else int(os.getenv("REVIEW_INITIAL_CANDIDATES", "30"))
        initial_top_k = initial_top_k if initial_top_k is not None else int(os.getenv("REVIEW_INITIAL_TOP_K", "20"))
        supplement_candidates = supplement_candidates if supplement_candidates is not None else int(os.getenv("REVIEW_SUPPLEMENT_CANDIDATES", "15"))
        supplement_top_k = supplement_top_k if supplement_top_k is not None else int(os.getenv("REVIEW_SUPPLEMENT_TOP_K", "10"))
        enable_expansion = enable_expansion if enable_expansion is not None else os.getenv("REVIEW_ENABLE_EXPANSION", "True").lower() == "true"
        
        start_time = time.time()
        logger.info(f"Starting iterative review generation")
        
        stats = {
            "query": user_question,
            "model": self.model,
            "max_iterations": max_iterations,
            "threshold": min_score_threshold,
            "iterations": 0,
            "final_score": 0,
            "adequate": False,
            "iteration_details": [],
            "all_documents": []  # Collect all documents used
        }
        
        # Initial retrieval and review
        current_review, initial_docs, initial_stats = self.initial_retrieval_and_review(
            user_question,
            initial_candidates,
            initial_top_k,
            enable_expansion
        )
        
        stats["initial_stats"] = initial_stats
        stats["total_documents"] = len(initial_docs)
        stats["all_documents"].extend(initial_docs)  # Add initial documents
        
        # Iterative improvement
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{max_iterations}")
            
            iter_start = time.time()
            
            # Audit review
            adequate, score, missing_aspects, reason = self.audit_review(
                user_question,
                current_review
            )
            
            iter_stats = {
                "iteration": iteration,
                "score": score,
                "adequate": adequate,
                "missing_aspects": missing_aspects,
                "time": 0,
                "new_docs": 0
            }
            
            logger.info(f"Audit: score={score}/10, adequate={adequate}")
            
            # Check if adequate
            if adequate and score >= min_score_threshold:
                logger.info(f"Review adequate (score {score} >= {min_score_threshold})")
                iter_stats["time"] = time.time() - iter_start
                stats["iteration_details"].append(iter_stats)
                stats["final_score"] = score
                stats["adequate"] = True
                break
            
            if not missing_aspects:
                logger.info(f"No missing aspects but score insufficient")
                iter_stats["time"] = time.time() - iter_start
                stats["iteration_details"].append(iter_stats)
                stats["final_score"] = score
                break
            
            # Supplementary retrieval
            logger.info(f"Supplementing {len(missing_aspects)} aspects")
            new_docs, supp_stats = self.supplementary_retrieval(
                missing_aspects,
                supplement_candidates,
                supplement_top_k
            )
            
            iter_stats["supplement_stats"] = supp_stats
            iter_stats["new_docs"] = len(new_docs)
            stats["total_documents"] += len(new_docs)
            stats["all_documents"].extend(new_docs)  # Add supplementary documents
            
            # Update review
            if new_docs:
                current_review = self.update_review(
                    user_question,
                    current_review,
                    new_docs,
                    missing_aspects
                )
                iter_stats["updated"] = True
            else:
                iter_stats["updated"] = False
            
            iter_stats["time"] = time.time() - iter_start
            stats["iteration_details"].append(iter_stats)
        
        # Final statistics
        stats["iterations"] = iteration
        stats["total_time"] = time.time() - start_time
        
        if iteration >= max_iterations:
            # Final audit
            final_adequate, final_score, _, _ = self.audit_review(
                user_question,
                current_review
            )
            stats["final_score"] = final_score
            stats["adequate"] = final_adequate
        
        logger.info(f"Completed: {iteration} iterations, score={stats['final_score']}/10, time={stats['total_time']:.2f}s")
        
        return current_review, stats
    
    # ==================== Display Method ====================
    def run_and_display(
        self,
        user_question: str,
        max_iterations: int = 3,
        min_score_threshold: int = 7,
        initial_candidates: int = 30,
        initial_top_k: int = 20,
        supplement_candidates: int = 15,
        supplement_top_k: int = 10,
        enable_expansion: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """Run iterative review generation and display results
        
        Args:
            user_question: User query
            max_iterations: Maximum iterations
            min_score_threshold: Minimum score threshold
            initial_candidates: Initial candidates
            initial_top_k: Initial top k
            supplement_candidates: Supplement candidates
            supplement_top_k: Supplement top k
            enable_expansion: Enable query expansion
            
        Returns:
            (final_review, statistics)
        """
        print("\n" + "="*80)
        print("ITERATIVE REVIEW GENERATION")
        print("="*80)
        print(f"Query: {user_question}")
        print(f"Model: {self.model}")
        print(f"Max iterations: {max_iterations}, Threshold: {min_score_threshold}")
        print(f"Query expansion: {'enabled' if enable_expansion else 'disabled'}")
        
        # Generate review
        final_review, stats = self.generate_iterative_review(
            user_question,
            max_iterations,
            min_score_threshold,
            initial_candidates,
            initial_top_k,
            supplement_candidates,
            supplement_top_k,
            enable_expansion
        )
        
        # Display statistics
        print("\n" + "="*80)
        print("STATISTICS")
        print("="*80)
        print(f"Total time: {stats['total_time']:.2f}s")
        print(f"Iterations: {stats['iterations']}/{stats['max_iterations']}")
        print(f"Final score: {stats['final_score']}/10")
        print(f"Adequate: {'Yes' if stats['adequate'] else 'No'}")
        print(f"Total documents: {stats['total_documents']}")
        
        # Expansion info
        exp_stats = stats.get('initial_stats', {}).get('expansion_stats', {})
        if exp_stats.get('success'):
            print(f"Query expansion: {exp_stats.get('ratio', 1.0):.1f}x")
        
        # Iteration details
        print("\nIteration details:")
        for iter_stat in stats['iteration_details']:
            print(f"  Iteration {iter_stat['iteration']}: score={iter_stat['score']}/10, "
                  f"new_docs={iter_stat.get('new_docs', 0)}, time={iter_stat.get('time', 0):.1f}s")
        
        # Display review
        print("\n" + "="*80)
        print("FINAL REVIEW")
        print("="*80)
        print(final_review)
        print("\n" + "="*80)
        
        return final_review, stats


if __name__ == "__main__":
    # Usage example - loads from .env by default
    system = IterativeReviewSystem()
    
    question = "What are the most effective nitrogen removal techniques in wastewater treatment?"
    review, stats = system.run_and_display(
        question,
        enable_expansion=True
    )