import numpy as np
from typing import List, Dict, Tuple, Set
import re
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading


class InformationDimension(Enum):
    """Information assessment dimensions - simplified to 4 core dimensions"""
    COVERAGE = "coverage"
    QUALITY = "quality"
    DIVERSITY = "diversity"
    RECENCY = "recency"


@dataclass
class TopicAnalysis:
    """Topic analysis result structure - simplified"""
    topic: str
    coverage_score: float
    quality_score: float
    diversity_score: float
    recency_score: float
    overall_score: float
    evidence_count: int
    source_diversity: int


class InformationDepthAssessor:
    """
    Optimized multi-dimensional information depth assessment system
    """

    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model
        self._embedding_cache = {}  # Cache for embeddings
        self._cache_lock = threading.Lock()

        # Consolidated quality indicators
        self.quality_indicators = {
            'high_quality': ['research', 'study', 'analysis', 'methodology', 'implementation',
                             'official', 'comprehensive', 'detailed', 'thorough'],
            'structure': ['1.', '2.', 'first', 'second', 'introduction', 'conclusion'],
            'references': ['reference', 'source', 'according to', 'research shows', 'studies indicate']
        }

        # Simplified authority sources
        self.authority_multipliers = {
            'edu': 1.3, 'gov': 1.3, 'org': 1.2, 'nature.com': 1.4, 'ieee.org': 1.4,
            'wikipedia': 1.1, 'blog': 0.9, 'forum': 0.8
        }

        # Regex patterns for efficient extraction
        self.year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        self.entity_patterns = [
            re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),  # Proper nouns
            re.compile(r'\b[A-Z]{2,}\b'),  # Acronyms
            re.compile(r'\b\w+(?:-\w+)+\b'),  # Hyphenated terms
        ]

        # Domain keywords for fast matching
        self.domain_keywords = {
            'neural', 'deep learning', 'machine learning', 'artificial intelligence',
            'software', 'hardware', 'system', 'platform', 'framework',
            'research', 'experiment', 'hypothesis', 'theory',
            'strategy', 'management', 'optimization', 'efficiency'
        }

    def assess_comprehensive_information_depth(
            self,
            query: str,
            sub_queries: List[str],
            all_results: List,
            global_state: dict
    ) -> Tuple[Dict, Dict]:
        """
        Optimized comprehensive information depth assessment
        """
        # Early exit for empty results
        if not all_results:
            return self._generate_empty_assessment(query), {}

        # Sample results if too many (performance optimization)
        sampled_results = self._sample_results(all_results, max_results=25)

        # Pre-compute all embeddings once
        self._precompute_embeddings(sampled_results)

        # Extract topics efficiently
        topics = self._extract_topics_fast(query, sub_queries)

        # Limit topics for performance
        if len(topics) > 6:
            topics = list(topics)[:6]

        # Parallel topic analysis
        topic_analyses = self._analyze_topics_parallel(topics, sampled_results, query)

        # Generate overall assessment
        overall_assessment = self._generate_overall_assessment(
            query, sub_queries, topic_analyses, sampled_results, global_state
        )

        return overall_assessment, topic_analyses

    def _sample_results(self, results: List, max_results: int = 25) -> List:
        """Sample results to limit processing time"""
        if len(results) <= max_results:
            return results

        # Keep first few and sample from the rest
        return results[:max_results // 2] + results[max_results // 2::len(results) // max_results][:max_results // 2]

    def _precompute_embeddings(self, results: List):
        """Pre-compute embeddings for all results to avoid repeated calculations"""
        for result in results:
            # Use first 400 characters for consistent embedding
            text_key = result.text[:400]
            if text_key not in self._embedding_cache:
                try:
                    embedding = self.embedding_model.embed_query(text_key)
                    with self._cache_lock:
                        self._embedding_cache[text_key] = embedding
                except:
                    # Fallback to zero vector
                    with self._cache_lock:
                        self._embedding_cache[text_key] = np.zeros(384)  # Assuming 384-dim embeddings

    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """Get cached embedding for text"""
        text_key = text[:400]
        with self._cache_lock:
            return self._embedding_cache.get(text_key, np.zeros(384))

    def _extract_topics_fast(self, query: str, sub_queries: List[str]) -> Set[str]:
        """Fast topic extraction using rule-based methods"""
        topics = set()
        all_queries = [query] + sub_queries

        for q in all_queries:
            # Extract using regex patterns
            for pattern in self.entity_patterns:
                matches = pattern.findall(q)
                topics.update(matches)

            # Extract domain keywords
            q_lower = q.lower()
            for keyword in self.domain_keywords:
                if keyword in q_lower:
                    topics.add(keyword)

            # Extract key phrases (2-3 words)
            words = q.split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i + 1]}"
                if len(bigram) > 6 and any(c.isupper() for c in bigram):
                    topics.add(bigram)

        # Remove very short topics and limit size
        topics = {t for t in topics if len(t) > 2}
        return topics

    def _analyze_topics_parallel(self, topics: Set[str], results: List, query: str) -> Dict[str, TopicAnalysis]:
        """Analyze topics in parallel for better performance"""
        topic_analyses = {}

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(4, len(topics))) as executor:
            future_to_topic = {
                executor.submit(self._analyze_topic_depth_fast, topic, results, query): topic
                for topic in topics
            }

            for future in future_to_topic:
                topic = future_to_topic[future]
                try:
                    topic_analyses[topic] = future.result()
                except Exception as e:
                    # Fallback analysis
                    topic_analyses[topic] = self._create_fallback_analysis(topic, results)

        return topic_analyses

    def _analyze_topic_depth_fast(self, topic: str, results: List, query: str) -> TopicAnalysis:
        """Fast topic analysis with simplified scoring"""
        # Filter relevant results efficiently
        relevant_results = self._filter_topic_relevant_results_fast(topic, results)

        # Calculate simplified scores
        coverage_score = self._calculate_coverage_score_fast(topic, relevant_results, query)
        quality_score = self._calculate_quality_score_fast(relevant_results)
        diversity_score = self._calculate_diversity_score_fast(relevant_results)
        recency_score = self._calculate_recency_score_fast(relevant_results)

        # Simplified weighted scoring
        weights = {'coverage': 0.3, 'quality': 0.35, 'diversity': 0.2, 'recency': 0.15}
        overall_score = (
                coverage_score * weights['coverage'] +
                quality_score * weights['quality'] +
                diversity_score * weights['diversity'] +
                recency_score * weights['recency']
        )

        return TopicAnalysis(
            topic=topic,
            coverage_score=coverage_score,
            quality_score=quality_score,
            diversity_score=diversity_score,
            recency_score=recency_score,
            overall_score=overall_score,
            evidence_count=len(relevant_results),
            source_diversity=len(set(r.metadata.get('source', 'unknown') for r in relevant_results))
        )

    def _filter_topic_relevant_results_fast(self, topic: str, results: List) -> List:
        """Fast filtering using cached embeddings and keyword matching"""
        if not results:
            return []

        relevant_results = []
        topic_words = set(topic.lower().split())

        for result in results:
            # Fast keyword matching first
            result_text_lower = result.text.lower()
            keyword_matches = sum(1 for word in topic_words if word in result_text_lower)

            if keyword_matches > 0:
                # Use semantic similarity only if keyword match exists
                try:
                    topic_embedding = self._get_cached_embedding(topic)
                    result_embedding = self._get_cached_embedding(result.text)

                    if np.any(topic_embedding) and np.any(result_embedding):
                        similarity = np.dot(topic_embedding, result_embedding)
                        if similarity > 0.25 or keyword_matches >= len(topic_words) * 0.6:
                            relevant_results.append(result)
                    else:
                        # Fallback to keyword matching
                        if keyword_matches >= len(topic_words) * 0.5:
                            relevant_results.append(result)
                except:
                    # Fallback to keyword matching
                    if keyword_matches >= len(topic_words) * 0.5:
                        relevant_results.append(result)

        return relevant_results

    def _calculate_coverage_score_fast(self, topic: str, results: List, query: str) -> float:
        """Fast coverage calculation"""
        if not results:
            return 0.0

        # Simplified expected aspects
        base_aspects = ['definition', 'example', 'application', 'benefit', 'method']
        covered_aspects = 0

        combined_text = ' '.join(result.text.lower() for result in results[:5])  # Limit for performance

        for aspect in base_aspects:
            if aspect in combined_text:
                covered_aspects += 1

        coverage_ratio = covered_aspects / len(base_aspects)
        volume_bonus = min(len(results) * 0.08, 0.25)

        return min(coverage_ratio + volume_bonus, 1.0)

    def _calculate_quality_score_fast(self, results: List) -> float:
        """Fast quality scoring with consolidated indicators"""
        if not results:
            return 0.0

        total_score = 0.0
        for result in results[:10]:  # Limit for performance
            text_lower = result.text.lower()
            score = 0.0

            # Check quality indicators
            for category, indicators in self.quality_indicators.items():
                matches = sum(1 for indicator in indicators if indicator in text_lower)
                if category == 'high_quality':
                    score += min(matches * 0.1, 0.4)
                elif category == 'structure':
                    score += min(matches * 0.05, 0.2)
                elif category == 'references':
                    score += min(matches * 0.05, 0.15)

            # Length-based quality (optimized)
            length_score = min(len(result.text) / 2000, 0.25)
            score += length_score

            # Authority bonus
            source = result.metadata.get('source', '').lower()
            for domain, multiplier in self.authority_multipliers.items():
                if domain in source:
                    score *= multiplier
                    break

            total_score += min(score, 1.0)

        return total_score / len(results) if results else 0.0

    def _calculate_diversity_score_fast(self, results: List) -> float:
        """Fast diversity calculation"""
        if not results:
            return 0.0

        # Source diversity
        sources = set(result.metadata.get('source', 'unknown') for result in results)
        source_diversity = min(len(sources) / 4, 1.0)

        # Content diversity (simplified)
        if len(results) > 1:
            # Sample pairs for efficiency
            sample_size = min(5, len(results))
            sampled_results = results[:sample_size]

            similarities = []
            for i in range(len(sampled_results)):
                for j in range(i + 1, len(sampled_results)):
                    try:
                        emb1 = self._get_cached_embedding(sampled_results[i].text)
                        emb2 = self._get_cached_embedding(sampled_results[j].text)
                        if np.any(emb1) and np.any(emb2):
                            sim = np.dot(emb1, emb2)
                            similarities.append(sim)
                    except:
                        continue

            content_diversity = 1.0 - np.mean(similarities) if similarities else 0.5
        else:
            content_diversity = 0.0

        return (source_diversity + content_diversity) / 2

    def _calculate_recency_score_fast(self, results: List) -> float:
        """Fast recency calculation using regex"""
        if not results:
            return 0.5

        from datetime import datetime
        current_year = datetime.now().year
        extracted_years = []

        for result in results[:10]:  # Limit for performance
            text = result.text[:800]  # Limit text length
            years = self.year_pattern.findall(text)

            for year_str in years:
                try:
                    year = int(year_str)
                    if 2000 <= year <= current_year:
                        extracted_years.append(year)
                except:
                    continue

        if not extracted_years:
            return 0.5

        avg_year = sum(extracted_years) / len(extracted_years)
        freshness = 1.0 - min((current_year - avg_year) / 8, 1.0)  # 8-year scale
        return max(0.0, min(freshness, 1.0))

    def _create_fallback_analysis(self, topic: str, results: List) -> TopicAnalysis:
        """Create fallback analysis when main analysis fails"""
        return TopicAnalysis(
            topic=topic,
            coverage_score=0.3,
            quality_score=0.3,
            diversity_score=0.3,
            recency_score=0.5,
            overall_score=0.35,
            evidence_count=len(results),
            source_diversity=min(len(set(r.metadata.get('source', 'unknown') for r in results)), 3)
        )

    def _generate_empty_assessment(self, query: str) -> Dict:
        """Generate assessment for empty results"""
        return {
            "needs_web_search": True,
            "confidence_score": 0.0,
            "knowledge_gaps": ["No relevant information found"],
            "suggested_search_queries": [query],
            "reasoning": "No information retrieved for analysis."
        }

    def _generate_overall_assessment(
            self,
            query: str,
            sub_queries: List[str],
            topic_analyses: Dict[str, TopicAnalysis],
            all_results: List,
            global_state: dict
    ) -> Dict:
        """Fast overall assessment generation"""
        if not topic_analyses:
            return self._generate_empty_assessment(query)

        # Calculate aggregate scores
        overall_scores = [analysis.overall_score for analysis in topic_analyses.values()]
        avg_score = np.mean(overall_scores)
        min_score = min(overall_scores)

        # Identify knowledge gaps efficiently
        knowledge_gaps = []
        low_scoring_topics = []

        for topic, analysis in topic_analyses.items():
            if analysis.overall_score < 0.4:
                low_scoring_topics.append(topic)
            if analysis.coverage_score < 0.3:
                knowledge_gaps.append(f"Limited coverage of {topic}")
            if analysis.quality_score < 0.3:
                knowledge_gaps.append(f"Low quality information about {topic}")

        # Simplified decision logic
        needs_web_search = (
                avg_score < 0.55 or
                min_score < 0.3 or
                len(all_results) < 3 or
                len(low_scoring_topics) > 2
        )

        # Generate search queries
        suggested_queries = []
        if needs_web_search:
            for topic in low_scoring_topics[:2]:  # Limit to 2 queries
                suggested_queries.append(f"{topic} comprehensive guide")

        # Simplified reasoning
        reasoning = f"Analysis: {avg_score:.2f} avg score, {len(all_results)} chunks, {len(knowledge_gaps)} gaps identified"

        return {
            "needs_web_search": needs_web_search,
            "confidence_score": avg_score,
            "knowledge_gaps": knowledge_gaps[:5],  # Limit gaps
            "suggested_search_queries": suggested_queries,
            "reasoning": reasoning,
            "detailed_scores": {
                topic: {
                    "overall": analysis.overall_score,
                    "coverage": analysis.coverage_score,
                    "quality": analysis.quality_score,
                    "evidence_count": analysis.evidence_count
                }
                for topic, analysis in list(topic_analyses.items())[:5]  # Limit detailed scores
            }
        }