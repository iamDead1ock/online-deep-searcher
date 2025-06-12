import time
import json
import asyncio
import requests
import numpy as np
from typing import List, Tuple, Dict

from deepsearcher.agent.deep_search import DeepSearch
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.utils import log,information_depth,web_query_deduplicator
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import BaseVectorDB, deduplicate_results
from deepsearcher.offline_loading import search_with_searxng

TOPIC_EXTRACTION_PROMPT = """
Extract ONLY the core technical topics and key concepts from the given query. Follow these rules:
1. Identify terms representing main subjects/entities
2. Include both broad domains and specific techniques when present
3. Maintain original technical terminology
4. NEVER combine unrelated concepts
5. PRESERVE exact capitalization and acronyms

OUTPUT REQUIREMENTS:
- ONLY output a valid Python list of strings
- No introductory text, explanations, or comments
- No markdown formatting (```python, ```, etc.)
- No trailing punctuation in list items
- Each string must be a standalone concept

Query: {query}

<EXAMPLE>
Example input:
Query: How do transformer architectures compare to CNN models in computer vision tasks, and what's the impact of attention mechanisms?

Example output:
["transformer architectures", "CNN models", "computer vision tasks", "attention mechanisms"]
</EXAMPLE>
"""

KNOWLEDGE_GAP_ASSESSMENT_PROMPT = """
Assess whether the retrieved information is sufficient to comprehensively answer the Original Query and its Sub-queries.

Original Query: {query}
Sub-queries: {sub_queries}

Retrieved Information Summary:
{retrieved_information_summary}

Number of relevant chunks found: {number_of_relevant_chunks_found}

Global Knowledge State:
- Covered Topics: {covered_topics}
- Information Depth: {information_depth}

OUTPUT REQUIREMENTS:
- ONLY output a valid JSON object.
- No introductory text, explanations, or comments.
- No markdown formatting (```json, ```, etc.).
- All keys and string values in the JSON object MUST be enclosed in double quotes.

The JSON object must contain the following keys:
- "needs_web_search": boolean (true if web search is needed, false otherwise)
- "confidence_score": float (0.0 to 1.0, higher is more confident in current knowledge)
- "knowledge_gaps": list of strings (specific areas where information is lacking)
- "suggested_search_queries": list of strings (new queries to fill gaps, ONLY if needs_web_search is true)
- "reasoning": string (explanation for the assessment, including why web search is or isn't needed)

<EXAMPLE>
Example input:
Query: What are the benefits and challenges of quantum computing for cryptography?
Sub-queries: ["benefits of quantum computing", "quantum computing challenges", "quantum cryptography"]
Retrieved Information Summary: "Some information on quantum computing basics and general cryptography was found, but specific details on quantum-safe algorithms are missing."
Number of relevant chunks found: 3
Global Knowledge State:
- Covered Topics: ['quantum computing', 'cryptography']
- Information Depth: {{'quantum computing': 0.6, 'cryptography': 0.4}}

Example output:
{{
  "needs_web_search": true,
  "confidence_score": 0.4,
  "knowledge_gaps": ["quantum-safe algorithms", "post-quantum cryptography standards", "specific implementation challenges of quantum cryptography"],
  "suggested_search_queries": ["post-quantum cryptography standards", "quantum resistant algorithms", "quantum cryptography implementation challenges"],
  "reasoning": "Current knowledge lacks depth on specific quantum-safe cryptographic algorithms and their implementation challenges, necessitating further web search."
}}
</EXAMPLE>

"""

WEB_SEARCH_QUERY_GENERATION_PROMPT = """
Based on the knowledge gaps identified, generate specific web search queries to fill these gaps.

Original Query: {query}
Knowledge Gaps: {gaps}
Current Information Coverage: {coverage_summary}

Generate 2-4 targeted search queries that would:
1. Fill the identified knowledge gaps
2. Provide current/recent information if needed  
3. Add diverse perspectives or sources
4. Complement existing retrieved information

Focus on queries that would yield high-quality, authoritative sources.

OUTPUT REQUIREMENTS:
- ONLY output a valid Python list of strings.
- No introductory text, explanations, or comments.
- No markdown formatting (```python, ```, etc.).
- No trailing punctuation in list items.
- Each string must be a standalone search query.

<EXAMPLE>
Example input:
Query: What are the primary benefits and implementation challenges of Retrieval-Augmented Generation (RAG) systems?
Knowledge Gaps: ["cost of RAG implementation in small businesses", "scalability issues of RAG for large datasets"]
Current Information Coverage: "Found general benefits and high-level challenges, but specific cost and scalability details for certain contexts are missing."

Expected output:
["RAG implementation cost small business", "scalability challenges Retrieval-Augmented Generation large datasets", "optimizing RAG cost efficiency", "RAG performance large scale data"]
</EXAMPLE>
"""

class OnlineDeepSearch(DeepSearch):
    """
    Online DeepSearch with dynamic web search capabilities.

    This agent first searches local knowledge base, assesses information gaps,
    and dynamically performs web searches when local information is insufficient.
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        max_iter: int = 3,
        route_collection: bool = True,
        text_window_splitter: bool = True,
        enable_web_search: bool = True,
        searxng_url: str = "http://localhost:8080",
        search_engines: List[str] = None,
        web_search_threshold: float = 0.6,
        max_web_searches_per_think: int = 2,
        max_web_results: int = 10,
        **kwargs,
    ):
        """
        Initialize the Enhanced DeepSearch agent.

        Args:
            enable_web_search: Whether to enable dynamic web search
            searxng_url: URL of the SearxNG instance
            search_engines: List of search engines to use
            web_search_threshold: Confidence threshold below which web search is triggered
            max_web_results: Maximum number of web search results to retrieve
        """
        super().__init__(llm, embedding_model, vector_db, max_iter, route_collection, text_window_splitter, **kwargs)

        self.enable_web_search = enable_web_search
        self.searxng_url = searxng_url
        self.search_engines = search_engines or ['google', 'bing', 'duckduckgo']
        self.web_search_threshold = web_search_threshold
        self.max_web_searches_per_think = max_web_searches_per_think
        self.max_web_results = max_web_results
        self.depth_assessor = information_depth.InformationDepthAssessor(self.embedding_model, self.llm)

        if self.enable_web_search:
            self._verify_searxng_availability()

    def _verify_searxng_availability(self) -> bool:
        """Verify that the SearxNG instance is available"""
        try:
            response = requests.get(
                f"{self.searxng_url}/search",
                params={'q': 'test', 'format': 'json'},
                timeout=10
            )
            if response.status_code == 200:
                log.color_print(f"<system> SearxNG instance verified at {self.searxng_url} </system>\n")
                return True
        except Exception as e:
            log.color_print(f"<system> SearxNG verification failed: {e} </system>\n")
            self.enable_web_search = False
        return False

    def _assess_comprehensive_knowledge_gaps(
            self,
            query: str,
            sub_queries: List[str],
            all_results: List[RetrievalResult],
            global_state: dict
    ) -> Tuple[dict, int]:
        """Comprehensively assess knowledge gaps, taking into account the global context"""

        # Perform comprehensive assessment using the assessor
        overall_assessment, topic_analyses = self.depth_assessor.assess_comprehensive_information_depth(
            query, sub_queries, all_results, global_state
        )

        # Update global state with information
        self._update_global_state_with_analysis(global_state, topic_analyses)

        # Log detailed analysis for debugging
        log.color_print(f"<assessment> Detailed topic analysis: </assessment>")
        for topic, analysis in topic_analyses.items():
            log.color_print(
                f"  - {topic}: Overall={analysis.overall_score:.3f}, Coverage={analysis.coverage_score:.3f}, Quality={analysis.quality_score:.3f}")

        # Token count estimation (since we're using LLM calls for topic extraction)
        estimated_tokens = len(query.split()) * 10 + len(sub_queries) * 20

        return overall_assessment, estimated_tokens

    def _update_global_state_with_analysis(self, global_state: dict, topic_analyses: Dict[str, information_depth.TopicAnalysis]):
        """Update global knowledge state with enhanced analysis results"""

        # Update covered topics
        global_state["covered_topics"].update(topic_analyses.keys())

        # Update information depth with comprehensive scores
        for topic, analysis in topic_analyses.items():
            global_state["information_depth"][topic] = {
                "overall_score": analysis.overall_score,
                "coverage": analysis.coverage_score,
                "quality": analysis.quality_score,
                "authority": analysis.authority_score,
                "diversity": analysis.diversity_score,
                "depth": analysis.depth_score,
                "coherence": analysis.coherence_score,
                "evidence_count": analysis.evidence_count,
                "source_diversity": analysis.source_diversity
            }

        # Identify remaining gaps based on multiple criteria
        remaining_gaps = []
        for topic, analysis in topic_analyses.items():
            if analysis.overall_score < 0.4:
                remaining_gaps.append(f"{topic} (overall low)")
            elif analysis.coverage_score < 0.3:
                remaining_gaps.append(f"{topic} (insufficient coverage)")
            elif analysis.quality_score < 0.3:
                remaining_gaps.append(f"{topic} (low quality)")

        global_state["remaining_gaps"] = remaining_gaps

        # Add comprehensive state tracking
        global_state["assessment_summary"] = {
            "total_topics": len(topic_analyses),
            "well_covered_topics": len([t for t, a in topic_analyses.items() if a.overall_score >= 0.7]),
            "partially_covered_topics": len([t for t, a in topic_analyses.items() if 0.4 <= a.overall_score < 0.7]),
            "poorly_covered_topics": len([t for t, a in topic_analyses.items() if a.overall_score < 0.4]),
            "average_coverage": np.mean([a.coverage_score for a in topic_analyses.values()]),
            "average_quality": np.mean([a.quality_score for a in topic_analyses.values()]),
            "total_evidence_count": sum(a.evidence_count for a in topic_analyses.values())
        }


    def _generate_web_search_queries(
            self,
            query: str,
            knowledge_gaps: List[str],
            coverage_summary: str
    ) -> Tuple[List[str], int]:
        """Generating web search queries based on knowledge gaps"""

        web_query_prompt = WEB_SEARCH_QUERY_GENERATION_PROMPT.format(
            query=query,
            gaps=knowledge_gaps,
            coverage_summary=coverage_summary
        )

        chat_response = self.llm.chat([{"role": "user", "content": web_query_prompt}])
        search_queries = self.llm.literal_eval(chat_response.content)

        return search_queries, chat_response.total_tokens

    def _update_global_knowledge_state(self, state: dict, query: str, results: List[RetrievalResult], source_type: str):
        """Updates the global knowledge state, tracking topics covered and depth of information"""
        if not results:
            return

        # Extract topic keywords
        query_topics = self._extract_topics_from_query(query)
        state["covered_topics"].update(query_topics)

        # Assess information depth
        for topic in query_topics:
            current_depth = state["information_depth"].get(topic, 0)
            new_depth = len([r for r in results if topic.lower() in r.text.lower()])
            state["information_depth"][topic] = max(current_depth, new_depth)

    def _extract_topics_from_query(self, query: str) -> set:
        """Extract core topics and key concepts from the query using the LLM."""
        topic_extraction_prompt = TOPIC_EXTRACTION_PROMPT.format(
            query=query
        )

        try:
            chat_response = self.llm.chat([{"role": "user", "content": topic_extraction_prompt}])

            topics_list = self.llm.literal_eval(chat_response.content)
            return set(topics_list)
        except Exception as e:
            log.color_print(
                f"<error> Error extracting topics with LLM: {e}. Falling back to simple word extraction. </error>\n")
            import re
            stop_words = {'what', 'how', 'why', 'when', 'where', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
            words = re.findall(r'\b\w+\b', query.lower())
            return {word for word in words if len(word) > 3 and word not in stop_words}

    def _deduplicate_web_queries(
            self,
            candidate_queries: List[str],
            performed_queries: List[str],
            performed_embeddings: List[np.ndarray]
    ) -> List[str]:
        """De-duplicate web search queries to avoid repeated searches"""

        if not hasattr(self, '_query_deduplicator'):
            self._query_deduplicator = web_query_deduplicator.WebQueryDeduplicator(self.llm, self.embedding_model)

        filtered_queries, reasons = self._query_deduplicator.smart_deduplicate(
            candidate_queries, performed_queries, performed_embeddings
        )

        for i, (query, reason) in enumerate(zip(candidate_queries, reasons)):
            if "Filtered" in reason:
                log.color_print(f"<dedup> {reason} </dedup>\n")
            else:
                log.color_print(f"<dedup> Query '{query}' - {reason} </dedup>\n")

        return filtered_queries

    def _process_web_metadata(self, web_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Process the metadata of web page results to ensure that it does not exceed the Milvus limit"""
        processed_results = []

        for result in web_results:
            if hasattr(result, 'metadata') and result.metadata:
                # Truncate metadata that is too long
                metadata_str = json.dumps(result.metadata, ensure_ascii=False)
                if len(metadata_str) > 60000:
                    truncated_metadata = {
                        "source": result.metadata.get("source",""),
                        "source_type": "web",
                        "title": result.metadata.get("title", "")[:500],
                    }

                    for key, value in result.metadata.items():
                        if key not in truncated_metadata and isinstance(value, str) and len(value) < 1000:
                            truncated_metadata[key] = value

                    result.metadata = truncated_metadata

            processed_results.append(result)

        return processed_results

    def _filter_urls_by_domain(self, urls: List[str]) -> List[str]:
        """
        Filter URLs to retain only academic or authoritative sources.

        This is particularly useful when using SearxNG, where many general-purpose
        engines (e.g., GitHub, blogs) are included in results.

        Academic domains are manually curated.
        """
        academic_domains = [
            "springer.com", "tandfonline.com", "jstor.org", "sciencedirect.com",
            "nature.com", "wiley.com", "cambridge.org", "oup.com", "sagepub.com",
            "semanticscholar.org", "arxiv.org", "core.ac.uk", "pnas.org",
            "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "researchgate.net",
        ]

        def is_academic(url: str) -> bool:
            return any(domain in url for domain in academic_domains)

        filtered = [url for url in urls if is_academic(url)]
        if not filtered:
            log.color_print("<web_filter> No academic URLs found. Fallback to original list. </web_filter>\n")
            return urls  # fallback if filter too strict
        return filtered

    async def _execute_batch_web_search(self, web_queries: List[str]) -> List[RetrievalResult]:
        """Batch execute network searches to optimize resource management"""
        all_web_results = []
        temp_collections = []

        try:
            # Collect all URLs
            all_urls = []
            for query in web_queries:
                log.color_print(f"<web_search> Searching web for: {query} </web_search>\n")
                urls = search_with_searxng(
                    query=query,
                    searxng_url=self.searxng_url,
                    search_engines=self.search_engines,
                    num_results=self.max_web_results // len(web_queries)
                )

                if urls:
                    log.color_print(f"<web_urls> URLs found for '{query}':")
                    for i, url in enumerate(urls, 1):
                        log.color_print(f"  {i}. {url}")
                    log.color_print("</web_urls>\n")
                else:
                    log.color_print(f"<web_urls> No URLs found for '{query}' </web_urls>\n")

                all_urls.extend(urls)

            # Remove duplicate URLs and print final list
            unique_urls = list(set(all_urls))

            # Filter academic sources
            filtered_urls = self._filter_urls_by_domain(unique_urls)
            log.color_print(f"<web_filter> Filtered academic URLs count: {len(filtered_urls)} </web_filter>\n")

            if filtered_urls:
                log.color_print("<web_urls_final> Final unique URLs to be processed:")
                for i, url in enumerate(filtered_urls, 1):
                    log.color_print(f"  {i}. {url}")
                log.color_print("</web_urls_final>\n")
            else:
                log.color_print(
                    "<web_urls_final> No filtered URLs passed, skipping web integration. </web_urls_final>\n")

            if filtered_urls:
                # Create temp collection and load
                temp_collection = f"web_search_batch_{int(time.time())}"
                temp_collections.append(temp_collection)

                log.color_print(
                    f"<web_integration> Loading web content into collection: {temp_collection} </web_integration>\n")

                # Batch load all URLs
                from deepsearcher.offline_loading import load_from_website
                await load_from_website(
                    urls=filtered_urls,  # << 👈 use filtered_urls here
                    collection_name=temp_collection,
                    collection_description=f"Batch web search results for queries: {', '.join(web_queries)}",
                    force_new_collection=True,
                    chunk_size=1500,
                    chunk_overlap=100
                )

                # Retrieving content from a temporary collection
                if temp_collection in self.vector_db.list_collections():
                    # Search using the combined vector of all queries
                    combined_query = " ".join(web_queries)
                    query_vector = self.embedding_model.embed_query(combined_query)

                    web_results = self.vector_db.search_data(
                        collection=temp_collection,
                        vector=query_vector,
                        limit=50
                    )
                    all_web_results.extend(web_results)

                    log.color_print(
                        f"<web_integration> Successfully loaded {len(unique_urls)} URLs, retrieved {len(web_results)} chunks from web content </web_integration>\n")

        except Exception as e:
            log.color_print(f"<web_integration> Error in batch web search: {e} </web_integration>\n")

        finally:
            # Clean up temporary collection
            for temp_collection in temp_collections:
                try:
                    if temp_collection in self.vector_db.list_collections():
                        self.vector_db.clear_db(temp_collection)
                        log.color_print(
                            f"<web_integration> Cleaned up temporary collection: {temp_collection} </web_integration>\n")
                except Exception as e:
                    log.color_print(f"<web_integration> Error cleaning up {temp_collection}: {e} </web_integration>\n")

                all_web_results = self._process_web_metadata(all_web_results)

        return all_web_results

    def _generate_comprehensive_retrieval_summary(self, all_results: List[RetrievalResult]) -> str:
        """Generate comprehensive search results summaries, including local and web results"""
        if not all_results:
            return "No information retrieved"

        # Sort results by source
        local_results = [r for r in all_results if not r.metadata.get("source_type") == "web"]
        web_results = [r for r in all_results if r.metadata.get("source_type") == "web"]

        summary_parts = []

        if local_results:
            local_sample = [result.text[:150] + "..." for result in local_results[:3]]
            summary_parts.append(f"Local Knowledge ({len(local_results)} chunks):")
            summary_parts.extend([f"- {text}" for text in local_sample])

        if web_results:
            web_sample = [result.text[:150] + "..." for result in web_results[:3]]
            summary_parts.append(f"Web Knowledge ({len(web_results)} chunks):")
            summary_parts.extend([f"- {text}" for text in web_sample])

        return "\n".join(summary_parts)

    async def async_retrieve(
            self,
            original_query: str,
            **kwargs
    ) -> Tuple[List[RetrievalResult], int, dict]:
        """
        Enhanced retrieve method with dynamic web search capability
        Modified to perform web search assessment after each individual local search
        """
        max_iter = kwargs.pop("max_iter", self.max_iter)
        min_info_gain_threshold = kwargs.pop("min_info_gain", 0.1)
        max_web_searches_per_think = kwargs.pop("max_web_searches_per_think", self.max_web_searches_per_think)
        web_search_similarity_threshold = kwargs.pop("web_search_similarity_threshold", 0.8)  # Search for deduplication threshold

        ### INITIALIZATION ###
        log.color_print(f"<query> {original_query} </query>\n")
        all_search_res = []  # Stores all unique retrieval results (local + web)
        all_sub_queries = []  # Stores all generated sub-queries
        total_tokens = 0

        performed_web_queries = []  # Keep track of actual web queries performed to avoid duplicates
        web_search_embeddings = []  # Embeddings of performed web queries
        accumulated_web_results = []  # Stores all web results accumulated across iterations

        global_knowledge_state = {
            "covered_topics": set(),
            "information_depth": {},
            "remaining_gaps": []
        }

        ### SUB QUERIES GENERATION ###
        sub_queries, used_token = self._generate_sub_queries(original_query)
        total_tokens += used_token

        if not sub_queries:
            log.color_print("No sub queries were generated by the LLM. Using original query for search.")
            if original_query and original_query.strip():
                sub_queries = [original_query]
            else:
                log.color_print("Original query is also empty. Exiting retrieval.")
                return all_search_res, total_tokens, {
                    "all_sub_queries": [],
                    "iterations_performed": 0,
                    "web_searches_performed": 0,
                    "web_results_count": 0
                }
        else:
            log.color_print(f"<think> Break down the original query into new sub queries: {sub_queries}</think>\n")

        all_sub_queries.extend(sub_queries)
        sub_gap_queries = [q for q in sub_queries if q and q.strip()]

        previous_iteration_results = []
        iterations_performed = 0
        total_web_searches = 0

        for iter in range(max_iter):
            iterations_performed = iter + 1
            log.color_print(f">> Iteration: {iterations_performed}\n")
            current_iteration_results = []

            if not sub_gap_queries:
                log.color_print(
                    f"<think> No valid sub-gap queries for iteration {iterations_performed}. Skipping search phase. </think>\n")
                break  # No more queries to process, exit loop
            else:
                # Process each query individually instead of parallel processing
                for query_idx, query in enumerate(sub_gap_queries, 1):
                    log.color_print(
                        f"<query_processing> Processing query {query_idx}/{len(sub_gap_queries)}: '{query}' </query_processing>\n")
                    query_web_searches = 0

                    # Perform individual local search
                    search_res, consumed_token = await self._search_chunks_from_vectordb(query, [query])
                    total_tokens += consumed_token
                    current_iteration_results.extend(search_res)

                    # Update global knowledge state for this query
                    self._update_global_knowledge_state(
                        global_knowledge_state,
                        query,
                        search_res,
                        "local"
                    )

                    local_results_count = len(current_iteration_results)
                    log.color_print(
                        f"<local_search_result> Retrieved {local_results_count} chunks from local knowledge base </local_search_result>\n")

                    # Assess web search need immediately after each local search
                    if self.enable_web_search and query_web_searches < max_web_searches_per_think:
                        comprehensive_assessment, assessment_tokens = self._assess_comprehensive_knowledge_gaps(
                            original_query,
                            all_sub_queries,
                            all_search_res + current_iteration_results + accumulated_web_results,
                            global_knowledge_state
                        )
                        total_tokens += assessment_tokens

                        log.color_print(
                            f"<think> Comprehensive knowledge assessment for '{query}': {comprehensive_assessment} </think>\n")

                        # Generate and execute web search if needed
                        if (comprehensive_assessment.get("needs_web_search", False) and
                                comprehensive_assessment.get("confidence_score", 1.0) < self.web_search_threshold):

                            candidate_web_queries, query_tokens = self._generate_web_search_queries(
                                original_query,
                                comprehensive_assessment.get("knowledge_gaps", []),
                                self._generate_comprehensive_retrieval_summary(
                                    all_search_res + current_iteration_results + accumulated_web_results
                                )
                            )
                            total_tokens += query_tokens

                            # Deduplication and filtering of web search queries
                            filtered_web_queries = self._deduplicate_web_queries(
                                candidate_web_queries,
                                performed_web_queries,
                                web_search_embeddings,
                                web_search_similarity_threshold
                            )

                            if filtered_web_queries:
                                remaining_search_budget = max_web_searches_per_think - query_web_searches
                                final_web_queries = filtered_web_queries[:remaining_search_budget]

                                log.color_print(
                                    f"<think> Executing {len(final_web_queries)} web searches for '{query}' (budget: {remaining_search_budget}): {final_web_queries} </think>\n")

                                # Perform web search immediately
                                web_results = await self._execute_batch_web_search(final_web_queries)

                                # Update search history
                                performed_web_queries.extend(final_web_queries)
                                for web_query in final_web_queries:
                                    query_embedding = self.embedding_model.embed_query(web_query)
                                    web_search_embeddings.append(query_embedding)

                                query_web_searches += len(final_web_queries)
                                total_web_searches += len(final_web_queries)
                                accumulated_web_results.extend(web_results)
                                current_iteration_results.extend(web_results)

                                # Update global knowledge state with web results
                                for web_query in final_web_queries:
                                    self._update_global_knowledge_state(
                                        global_knowledge_state,
                                        web_query,
                                        web_results,
                                        "web"
                                    )

                                log.color_print(
                                    f"<think> Web search for '{query}' added {len(web_results)} chunks. </think>\n")

                            else:
                                log.color_print(
                                    f"<think> No new web queries after deduplication for '{query}'. </think>\n")
                        else:
                            log.color_print(
                                f"<think> No web search needed for '{query}' based on assessment. </think>\n")

            current_iteration_results = deduplicate_results(current_iteration_results)

            info_gain = self._calculate_information_gain(
                previous_iteration_results if previous_iteration_results else all_search_res,
                current_iteration_results
            )

            log.color_print(f"<iteration> Information gain from iteration {iter + 1}: {info_gain:.4f} <iteration>\n")

            all_search_res.extend(current_iteration_results)
            all_search_res = deduplicate_results(all_search_res)

            previous_iteration_results = current_iteration_results

            if iter > 0 and info_gain < min_info_gain_threshold:
                log.color_print(
                    f"<think> Information gain below threshold ({info_gain:.4f} < {min_info_gain_threshold}). Stopping iterations. </think>\n")
                break

            if iterations_performed == max_iter:
                log.color_print("<think> Reached maximum iterations. Stopping. </think>\n")
                break

            log.color_print("<think> Reflecting on the search results... </think>\n")
            gap_queries_list, consumed_token = self._generate_gap_queries(
                original_query, all_sub_queries, all_search_res
            )
            total_tokens += consumed_token

            sub_gap_queries = [q for q in gap_queries_list if q and q.strip()]

            if not sub_gap_queries or len(sub_gap_queries) == 0:
                log.color_print("<think> No new search queries were generated. Exiting. </think>\n")
                break
            else:
                log.color_print(
                    f"<think> New search queries for next iteration: {sub_gap_queries} </think>\n"
                )
                all_sub_queries.extend(sub_gap_queries)
                all_sub_queries = list(dict.fromkeys(all_sub_queries))

        additional_info = {
            "all_sub_queries": all_sub_queries,
            "iterations_performed": iterations_performed,
            "total_web_searches": total_web_searches,
            "web_results_count": len(accumulated_web_results)
        }

        return all_search_res, total_tokens, additional_info