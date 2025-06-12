import log
import numpy as np
import re
from typing import List, Tuple, Set
from functools import lru_cache

SEMANTIC_COMPARISON_PROMPT = """
Compare these two search queries and determine if they are essentially asking for the same information:

Query 1: "{query1}"
Query 2: "{query2}"

Consider:
1. Do they seek the same type of information?
2. Do they have the same search intent and focus?
3. Would they likely return substantially overlapping results?
4. Are the key concepts and entities the same?

Respond with:
- "DUPLICATE" if they are essentially the same query
- "DIFFERENT" if they seek different information or have different intents
- Provide a brief reason for your decision

Format: DECISION|REASON
"""

class WebQueryDeduplicator:
    """
    web query deduplicator using multi-layered strategies to avoid excessive deduplication.
    """

    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model

        # Similarity thresholds
        self.thresholds = {
            'exact_match': 1.0,
            'semantic_high': 0.95,
            'semantic_medium': 0.85,
            'semantic_low': 0.75
        }

        # Intent keyword categories
        self.intent_keywords = {
            'temporal': ['recent', 'latest', 'current', 'new', 'emerging', 'future', 'historical', 'past', 'evolution'],
            'analytical': ['findings', 'results', 'analysis', 'insights', 'conclusions', 'outcomes'],
            'methodological': ['approaches', 'methods', 'techniques', 'strategies', 'frameworks'],
            'applicational': ['applications', 'implementations', 'use cases', 'deployment', 'practice'],
            'comparative': ['comparison', 'versus', 'vs', 'compared', 'differences', 'similarities'],
            'evaluative': ['benefits', 'challenges', 'advantages', 'disadvantages', 'limitations', 'effectiveness'],
            'taxonomical': ['types', 'categories', 'classification', 'taxonomy', 'kinds'],
            'causal': ['impact', 'effects', 'influence', 'consequences', 'causes', 'reasons']
        }

    def extract_query_intent(self, query: str) -> Set[str]:
        """
        Extract high-level semantic intent categories from a query.
        """
        query_lower = query.lower()
        intents = set()
        for intent_type, keywords in self.intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                intents.add(intent_type)
        return intents

    def extract_key_entities(self, query: str) -> Set[str]:
        """
        Extract significant keywords/entities from a query.
        """
        stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
                      'what', 'how', 'why', 'when', 'where', 'who', 'which', 'and', 'or', 'but'}
        words = re.findall(r'\b\w+\b', query.lower())
        return {word for word in words if len(word) > 2 and word not in stop_words}

    def calculate_overlap(self, s1: Set[str], s2: Set[str]) -> float:
        """
        Generic Jaccard similarity for two sets.
        """
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    def calculate_intent_overlap(self, query1: str, query2: str) -> float:
        return self.calculate_overlap(self.extract_query_intent(query1), self.extract_query_intent(query2))

    def calculate_entity_overlap(self, query1: str, query2: str) -> float:
        return self.calculate_overlap(self.extract_key_entities(query1), self.extract_key_entities(query2))

    @lru_cache(maxsize=1000)
    def llm_semantic_comparison(self, query1: str, query2: str) -> Tuple[bool, str]:
        """
        Use LLM to semantically compare two queries and decide if they are duplicates.
        """
        semantic_comparison_prompt=SEMANTIC_COMPARISON_PROMPT.format(
            query1=query1,
            query2=query2
        )
        try:
            response = self.llm.chat([{"role": "user", "content": semantic_comparison_prompt}])
            content = response.content.strip()
            if '|' in content:
                decision, reason = content.split('|', 1)
                return decision.strip().upper() == 'DUPLICATE', reason.strip()
            return 'DUPLICATE' in content.upper(), content
        except Exception as e:
            return False, f"LLM comparison failed: {e}"

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        """
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def smart_deduplicate(
            self,
            candidate_queries: List[str],
            performed_queries: List[str],
            performed_embeddings: List[np.ndarray]
    ) -> Tuple[List[str], List[str]]:
        """
        Smart deduplication logic combining embedding, intent/entity overlap, and LLM.
        Returns (filtered queries, reasons).
        """
        if not candidate_queries:
            return [], []

        filtered_queries = []
        dedup_reasons = []

        for candidate in candidate_queries:
            should_keep = True
            skip_reason = ""
            candidate_embedding = self.embedding_model.embed_query(candidate)

            for i, performed_embedding in enumerate(performed_embeddings):
                similarity = self.cosine_similarity(candidate_embedding, performed_embedding)
                performed_query = performed_queries[i]

                if similarity >= self.thresholds['exact_match']:
                    should_keep = False
                    skip_reason = f"Exact match with performed query: '{performed_query}'"
                    break
                elif similarity >= self.thresholds['semantic_high']:
                    if self.calculate_intent_overlap(candidate, performed_query) > 0.8 and \
                       self.calculate_entity_overlap(candidate, performed_query) > 0.8:
                        should_keep = False
                        skip_reason = f"High semantic similarity with overlapping intent/entities: '{performed_query}'"
                        break
                elif similarity >= self.thresholds['semantic_medium']:
                    is_duplicate, reason = self.llm_semantic_comparison(candidate, performed_query)
                    if is_duplicate:
                        should_keep = False
                        skip_reason = f"LLM flagged as duplicate of '{performed_query}': {reason}"
                        break

            if should_keep:
                for existing in filtered_queries:
                    existing_embedding = self.embedding_model.embed_query(existing)
                    similarity = self.cosine_similarity(candidate_embedding, existing_embedding)
                    if similarity >= self.thresholds['semantic_high']:
                        if self.calculate_intent_overlap(candidate, existing) > 0.8 and \
                           self.calculate_entity_overlap(candidate, existing) > 0.8:
                            should_keep = False
                            skip_reason = f"High similarity with existing batch query: '{existing}'"
                            break
                    elif similarity >= self.thresholds['semantic_medium']:
                        is_duplicate, reason = self.llm_semantic_comparison(candidate, existing)
                        if is_duplicate:
                            should_keep = False
                            skip_reason = f"LLM flagged as duplicate of batch query '{existing}': {reason}"
                            break

            if should_keep:
                filtered_queries.append(candidate)
                dedup_reasons.append("Kept - sufficiently different")
            else:
                dedup_reasons.append(f"Filtered - {skip_reason}")

        return filtered_queries, dedup_reasons
