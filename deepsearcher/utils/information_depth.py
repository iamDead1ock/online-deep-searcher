import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import re
from dataclasses import dataclass
from enum import Enum

EXTRACT_TOPICS_PROMPT = """
Extract key topics, concepts, and entities from this query. Focus on:
1. Technical terms and concepts
2. Domain-specific terminology  
3. Named entities (people, organizations, technologies)
4. Abstract concepts that need explanation

Query: {query}

Return only a Python list of strings, no other text.
"""

EXTRACT_YEAR_PROMPT = """
Extract all years mentioned in the following text.{text}

Return a Python list of integers only.
"""

class InformationDimension(Enum):
    """Information assessment dimensions"""
    COVERAGE = "coverage"
    QUALITY = "quality"
    AUTHORITY = "authority"
    RECENCY = "recency"
    DIVERSITY = "diversity"
    DEPTH = "depth"
    COHERENCE = "coherence"


@dataclass
class TopicAnalysis:
    """Topic analysis result structure"""
    topic: str
    coverage_score: float
    quality_score: float
    authority_score: float
    recency_score: float
    diversity_score: float
    depth_score: float
    coherence_score: float
    overall_score: float
    evidence_count: int
    source_diversity: int


class InformationDepthAssessor:
    """
    Multi-dimensional information depth assessment system
    """

    def __init__(self, embedding_model, llm):
        self.embedding_model = embedding_model
        self.llm = llm

        # Quality indicators for different content types
        self.quality_indicators = {
            'academic': ['research', 'study', 'analysis', 'methodology', 'findings', 'conclusion'],
            'technical': ['implementation', 'architecture', 'algorithm', 'performance', 'optimization'],
            'authoritative': ['official', 'government', 'institution', 'organization', 'standard'],
            'comprehensive': ['overview', 'comprehensive', 'complete', 'detailed', 'thorough']
        }

        # Authority indicators
        self.authority_sources = {
            'high': ['edu', 'gov', 'org', 'nature.com', 'ieee.org', 'acm.org'],
            'medium': ['com', 'net', 'wikipedia'],
            'low': ['blog', 'forum', 'social']
        }

    def assess_comprehensive_information_depth(
            self,
            query: str,
            sub_queries: List[str],
            all_results: List,  # RetrievalResult objects
            global_state: dict
    ) -> Tuple[Dict, Dict]:
        """
        Comprehensive multi-dimensional information depth assessment

        Returns:
            Tuple of (overall_assessment, detailed_topic_analysis)
        """

        # Extract and analyze topics
        topics = self._extract_enhanced_topics(query, sub_queries)

        # Perform multi-dimensional analysis for each topic
        topic_analyses = {}
        for topic in topics:
            topic_analyses[topic] = self._analyze_topic_depth(topic, all_results, query)

        # Generate overall assessment
        overall_assessment = self._generate_overall_assessment(
            query, sub_queries, topic_analyses, all_results, global_state
        )

        return overall_assessment, topic_analyses

    def _extract_enhanced_topics(self, query: str, sub_queries: List[str]) -> Set[str]:
        """
        Enhanced topic extraction using multiple strategies
        """
        topics = set()

        # Strategy 1: LLM-based extraction
        all_queries = [query] + sub_queries
        for q in all_queries:
            llm_topics = self._extract_topics_with_llm(q)
            topics.update(llm_topics)

        # Strategy 2: Named entity recognition patterns
        ner_topics = self._extract_entities_with_patterns(query)
        topics.update(ner_topics)

        # Strategy 3: Domain-specific keywords
        domain_topics = self._extract_domain_keywords(query)
        topics.update(domain_topics)

        return topics

    def _extract_topics_with_llm(self, query: str) -> Set[str]:
        """Extract topics using LLM with enhanced prompt"""

        extract_topics_prompt = EXTRACT_TOPICS_PROMPT.format(
            query=query,
        )

        try:
            response = self.llm.chat([{"role": "user", "content": extract_topics_prompt}])
            topics = self.llm.literal_eval(response.content)
            return set(topics) if isinstance(topics, list) else set()
        except:
            return set()

    def _extract_entities_with_patterns(self, query: str) -> Set[str]:
        """Extract entities using regex patterns"""
        entities = set()

        # Common entity patterns
        patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper nouns (e.g., "Machine Learning")
            r'\b[A-Z]{2,}\b',  # Acronyms (e.g., "AI", "ML", "CNN")
            r'\b\w+(?:-\w+)+\b',  # Hyphenated terms
            r'\b\w*[Tt]echnology\b|\b\w*[Aa]lgorithm\b|\b\w*[Mm]ethod\b'  # Technical suffixes
        ]

        for pattern in patterns:
            matches = re.findall(pattern, query)
            entities.update(matches)

        return entities

    def _extract_domain_keywords(self, query: str) -> Set[str]:
        """Extract domain-specific keywords"""
        domain_keywords = {
            'ai_ml': ['neural', 'deep learning', 'machine learning', 'artificial intelligence'],
            'tech': ['software', 'hardware', 'system', 'platform', 'framework'],
            'science': ['research', 'experiment', 'hypothesis', 'theory'],
            'business': ['strategy', 'management', 'optimization', 'efficiency']
        }

        found_keywords = set()
        query_lower = query.lower()

        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    found_keywords.add(keyword)

        return found_keywords

    def _analyze_topic_depth(self, topic: str, results: List, query: str) -> TopicAnalysis:
        """
        Multi-dimensional analysis of topic information depth
        """

        # Filter relevant results for this topic
        relevant_results = self._filter_topic_relevant_results(topic, results)

        # Calculate each dimension
        coverage_score = self._calculate_coverage_score(topic, relevant_results, query)
        quality_score = self._calculate_quality_score(relevant_results)
        authority_score = self._calculate_authority_score(relevant_results)
        recency_score = self._calculate_recency_score(relevant_results)
        diversity_score = self._calculate_diversity_score(relevant_results)
        depth_score = self._calculate_content_depth_score(relevant_results, topic)
        coherence_score = self._calculate_coherence_score(relevant_results, topic)

        # Calculate weighted overall score
        weights = {
            'coverage': 0.20, 'quality': 0.18, 'authority': 0.15,
            'recency': 0.10, 'diversity': 0.12, 'depth': 0.15, 'coherence': 0.10
        }

        overall_score = (
                coverage_score * weights['coverage'] +
                quality_score * weights['quality'] +
                authority_score * weights['authority'] +
                recency_score * weights['recency'] +
                diversity_score * weights['diversity'] +
                depth_score * weights['depth'] +
                coherence_score * weights['coherence']
        )

        return TopicAnalysis(
            topic=topic,
            coverage_score=coverage_score,
            quality_score=quality_score,
            authority_score=authority_score,
            recency_score=recency_score,
            diversity_score=diversity_score,
            depth_score=depth_score,
            coherence_score=coherence_score,
            overall_score=overall_score,
            evidence_count=len(relevant_results),
            source_diversity=len(set(r.metadata.get('source', 'unknown') for r in relevant_results))
        )

    def _filter_topic_relevant_results(self, topic: str, results: List) -> List:
        """Filter results relevant to specific topic using semantic similarity"""
        if not results:
            return []

        topic_embedding = self.embedding_model.embed_query(topic)
        relevant_results = []

        for result in results:
            # Calculate semantic similarity
            try:
                result_embedding = self.embedding_model.embed_query(result.text[:500])  # First 500 chars
                similarity = np.dot(topic_embedding, result_embedding)

                # Also check for keyword presence
                keyword_match = any(
                    word.lower() in result.text.lower()
                    for word in topic.split()
                )

                if similarity > 0.3 or keyword_match:  # Threshold for relevance
                    relevant_results.append(result)
            except:
                # Fallback to keyword matching
                if any(word.lower() in result.text.lower() for word in topic.split()):
                    relevant_results.append(result)

        return relevant_results

    def _calculate_coverage_score(self, topic: str, results: List, query: str) -> float:
        """Calculate topic coverage completeness"""
        if not results:
            return 0.0

        # Generate expected subtopics for comprehensive coverage
        expected_aspects = self._generate_expected_aspects(topic, query)
        covered_aspects = 0

        for aspect in expected_aspects:
            if any(aspect.lower() in result.text.lower() for result in results):
                covered_aspects += 1

        coverage_ratio = covered_aspects / len(expected_aspects) if expected_aspects else 0

        # Bonus for information volume
        volume_bonus = min(len(results) * 0.1, 0.3)

        return min(coverage_ratio + volume_bonus, 1.0)

    def _generate_expected_aspects(self, topic: str, query: str) -> List[str]:
        """Generate expected aspects for comprehensive topic coverage"""

        # Common aspects for different topic types
        base_aspects = ['definition', 'examples', 'applications', 'benefits', 'challenges']

        # Topic-specific aspects
        if any(tech_term in topic.lower() for tech_term in ['algorithm', 'model', 'system']):
            base_aspects.extend(['implementation', 'performance', 'comparison'])
        elif 'learning' in topic.lower():
            base_aspects.extend(['methodology', 'training', 'evaluation'])
        elif any(business_term in query.lower() for business_term in ['strategy', 'management']):
            base_aspects.extend(['costs', 'ROI', 'risks'])

        return base_aspects

    def _calculate_quality_score(self, results: List) -> float:
        """Calculate information quality based on multiple indicators"""
        if not results:
            return 0.0

        quality_scores = []

        for result in results:
            text = result.text.lower()
            score = 0.0

            # Check for quality indicators
            for category, indicators in self.quality_indicators.items():
                matches = sum(1 for indicator in indicators if indicator in text)
                score += matches * 0.1

            # Check for structured content
            if any(marker in text for marker in ['1.', '2.', 'first', 'second', 'conclusion']):
                score += 0.2

            # Check for citations/references
            if any(ref in text for ref in ['reference', 'source', 'study', 'research']):
                score += 0.15

            # Length-based quality (moderate length often indicates thorough coverage)
            length_score = min(len(text) / 1000, 1.0) * 0.2
            score += length_score

            quality_scores.append(min(score, 1.0))

        return np.mean(quality_scores) if quality_scores else 0.0

    def _calculate_authority_score(self, results: List) -> float:
        """Calculate source authority and credibility"""
        if not results:
            return 0.0

        authority_scores = []

        for result in results:
            score = 0.5  # Base score
            source = result.metadata.get('source', '').lower()

            # Check source authority
            if any(domain in source for domain in self.authority_sources['high']):
                score += 0.4
            elif any(domain in source for domain in self.authority_sources['medium']):
                score += 0.2
            elif any(domain in source for domain in self.authority_sources['low']):
                score -= 0.1

            # Check for authoritative language
            text = result.text.lower()
            if any(phrase in text for phrase in ['according to', 'research shows', 'studies indicate']):
                score += 0.1

            authority_scores.append(max(0.0, min(score, 1.0)))

        return np.mean(authority_scores) if authority_scores else 0.5

    def _calculate_recency_score(self, results: List) -> float:
        """Calculate information recency (when available)"""
        if not results:
            return 0.5  # Neutral score when no date info

        from datetime import datetime
        current_year = datetime.now().year

        extracted_years = []

        for result in results:
            try:
                text = result.text[:1000]
                # Call LLM to extract the year list
                extract_year_prompt=EXTRACT_YEAR_PROMPT.format(
                    text = text,
                )

                response = self.llm.chat([
                    {
                        "role": "user",
                        "content": extract_year_prompt
                    }
                ])
                years = self.llm.literal_eval(response.content)
                if isinstance(years, list):
                    filtered = [y for y in years if 2000 <= y <= current_year]
                    extracted_years.extend(filtered)
            except:
                continue

        if not extracted_years:
            return 0.5

        avg_year = sum(extracted_years) / len(extracted_years)
        freshness = 1.0 - min((current_year - avg_year) / 10, 1.0)
        return round(freshness, 3)

    def _calculate_diversity_score(self, results: List) -> float:
        """Calculate source and perspective diversity"""
        if not results:
            return 0.0

        # Source diversity
        sources = set(result.metadata.get('source', 'unknown') for result in results)
        source_diversity = min(len(sources) / 5, 1.0)  # Normalize to max 5 sources

        # Content diversity (using embedding similarity)
        if len(results) > 1:
            similarities = []
            for i in range(min(len(results), 5)):  # Sample for efficiency
                for j in range(i + 1, min(len(results), 5)):
                    try:
                        emb1 = self.embedding_model.embed_query(results[i].text[:300])
                        emb2 = self.embedding_model.embed_query(results[j].text[:300])
                        sim = np.dot(emb1, emb2)
                        similarities.append(sim)
                    except:
                        continue

            content_diversity = 1.0 - np.mean(similarities) if similarities else 0.5
        else:
            content_diversity = 0.0

        return (source_diversity + content_diversity) / 2

    def _calculate_content_depth_score(self, results: List, topic: str) -> float:
        """Calculate the depth of content analysis"""
        if not results:
            return 0.0

        depth_scores = []

        for result in results:
            text = result.text
            score = 0.0

            # Check for analytical depth indicators
            depth_indicators = [
                'because', 'therefore', 'however', 'furthermore', 'analysis',
                'explanation', 'reason', 'cause', 'effect', 'implication'
            ]

            matches = sum(1 for indicator in depth_indicators if indicator in text.lower())
            score += min(matches * 0.1, 0.5)

            # Check for detailed explanations (longer paragraphs)
            sentences = text.split('.')
            long_sentences = [s for s in sentences if len(s.strip()) > 100]
            score += min(len(long_sentences) * 0.05, 0.3)

            # Check for technical detail
            if any(term in text.lower() for term in ['method', 'process', 'step', 'procedure']):
                score += 0.2

            depth_scores.append(min(score, 1.0))

        return np.mean(depth_scores) if depth_scores else 0.0

    def _calculate_coherence_score(self, results: List, topic: str) -> float:
        """Calculate logical coherence and consistency across sources"""
        if len(results) < 2:
            return 0.5  # Neutral score for single source

        # Sample results for efficiency
        sample_results = results[:5] if len(results) > 5 else results

        try:
            # Calculate semantic coherence using embeddings
            embeddings = []
            for result in sample_results:
                emb = self.embedding_model.embed_query(result.text[:400])
                embeddings.append(emb)

            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j])
                    similarities.append(sim)

            coherence = np.mean(similarities) if similarities else 0.5

            # Adjust for topic consistency
            topic_mentions = sum(
                1 for result in sample_results
                if any(word.lower() in result.text.lower() for word in topic.split())
            )
            topic_consistency = topic_mentions / len(sample_results)

            return (coherence + topic_consistency) / 2

        except:
            return 0.5

    def _generate_overall_assessment(
            self,
            query: str,
            sub_queries: List[str],
            topic_analyses: Dict[str, TopicAnalysis],
            all_results: List,
            global_state: dict
    ) -> Dict:
        """Generate comprehensive overall assessment"""

        if not topic_analyses:
            return {
                "needs_web_search": True,
                "confidence_score": 0.0,
                "knowledge_gaps": ["No relevant information found"],
                "suggested_search_queries": [query],
                "reasoning": "No information retrieved for any identified topics."
            }

        # Calculate aggregate scores
        overall_scores = [analysis.overall_score for analysis in topic_analyses.values()]
        avg_score = np.mean(overall_scores)
        min_score = min(overall_scores)

        # Identify knowledge gaps
        knowledge_gaps = []
        low_scoring_topics = []

        for topic, analysis in topic_analyses.items():
            if analysis.overall_score < 0.4:
                low_scoring_topics.append(topic)

            if analysis.coverage_score < 0.3:
                knowledge_gaps.append(f"Insufficient coverage of {topic}")
            if analysis.quality_score < 0.3:
                knowledge_gaps.append(f"Low quality information about {topic}")
            if analysis.diversity_score < 0.3:
                knowledge_gaps.append(f"Limited perspectives on {topic}")

        # Determine web search necessity
        needs_web_search = (
                avg_score < 0.6 or
                min_score < 0.3 or
                len(all_results) < 3 or
                len(knowledge_gaps) > 2
        )

        # Generate search queries for gaps
        suggested_queries = []
        if needs_web_search:
            for topic in low_scoring_topics[:3]:  # Limit to 3 queries
                suggested_queries.append(f"{topic} comprehensive guide")

        # Generate reasoning
        reasoning_parts = []
        reasoning_parts.append(f"Average topic coverage score: {avg_score:.2f}")
        reasoning_parts.append(f"Retrieved {len(all_results)} total chunks")

        if knowledge_gaps:
            reasoning_parts.append(f"Identified {len(knowledge_gaps)} knowledge gaps")
        if low_scoring_topics:
            reasoning_parts.append(f"Topics needing improvement: {', '.join(low_scoring_topics[:3])}")

        return {
            "needs_web_search": needs_web_search,
            "confidence_score": avg_score,
            "knowledge_gaps": knowledge_gaps,
            "suggested_search_queries": suggested_queries,
            "reasoning": ". ".join(reasoning_parts),
            "detailed_scores": {
                topic: {
                    "overall": analysis.overall_score,
                    "coverage": analysis.coverage_score,
                    "quality": analysis.quality_score,
                    "authority": analysis.authority_score,
                    "diversity": analysis.diversity_score,
                    "evidence_count": analysis.evidence_count
                }
                for topic, analysis in topic_analyses.items()
            }
        }