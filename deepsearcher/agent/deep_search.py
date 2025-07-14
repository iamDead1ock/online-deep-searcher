import asyncio
import numpy as np
from typing import List, Tuple, Dict, Any

from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.utils import log,ReportQualityEnhancer
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import BaseVectorDB, deduplicate_results

CONTENT_ARCHITECTURE_PROMPT = """As an expert content strategist and domain analyst, analyze the query and retrieved information to create an optimal content architecture plan with enhanced structural sophistication.

Query: {query}
Sub-queries: {sub_queries}
Available Information Summary: {info_summary}
Query Characteristics: {query_characteristics}

Based on the query type and available information, design a comprehensive content architecture that includes:

1. Content Type Classification with Confidence Score:
   - Comparative analysis (with specific comparison dimensions)
   - Causal relationship analysis (with cause-effect chains)
   - Historical development review (with temporal markers)
   - Technical explanation (with complexity level)
   - Policy analysis (with stakeholder perspectives)
   - Market analysis (with trend identification)
   - Other (specify with reasoning)

2. Advanced Logical Structure Plan:
   - Sequential (chronological/step-by-step with timeline)
   - Categorical (thematic grouping with cross-references)
   - Hierarchical (general to specific with depth levels)
   - Comparative (side-by-side with evaluation criteria)
   - Problem-solution framework (with alternatives analysis)
   - Matrix structure (multi-dimensional analysis)
   - Hybrid approach (combination with transition logic)

3. Detailed Content Outline with Quality Indicators:
   - Main sections with estimated word counts
   - Key arguments and supporting evidence allocation
   - Critical analysis points and counterarguments
   - Data integration strategies and visualization suggestions
   - Cross-reference and coherence maintenance plans

4. Professional Writing Standards:
   - Target audience level (academic, professional, general)
   - Required depth of analysis and evidence standards
   - Citation and referencing requirements
   - Technical terminology usage guidelines
   - Objectivity vs. advocacy balance

Return your response in the following JSON format:
{{
    "content_type": "string",
    "confidence_score": 0.0-1.0,
    "structure_pattern": "string",
    "target_audience": "string",
    "estimated_length": "string",
    "outline": [
        {{
            "section": "string",
            "subsections": ["string"],
            "key_arguments": ["string"],
            "evidence_strategy": "string",
            "critical_points": ["string"],
            "estimated_words": 0
        }}
    ],
    "quality_criteria": {{
        "analysis_depth": "string",
        "evidence_requirements": "string",
        "critical_thinking": "string",
        "technical_level": "string",
        "objectivity_level": "string"
    }},
    "visualization_suggestions": ["string"],
    "cross_references": ["string"]
}}
"""

SUB_QUERY_PROMPT = """In order to answer this question more comprehensively, please break down the original question into at most four sub-questions. 
I hope you will not simply split the original question directly, but after summarizing the original question, get at most four sub-questions related to the direction of the original question.
These sub-questions can cover the core elements of the original question very well, and each sub-question has the value of independent exploration.Also, sub-questions should be general and not too detailed.
Return as list of str.If this is a very simple question and no decomposition is necessary, then keep the only one original question in the python code list.

Original Question: {original_query}


<EXAMPLE>
Example input:
"Explain deep learning"

Example output:
[
    "What is deep learning and what are its core principles?",
    "How does deep learning compare to other machine learning approaches?",
    "What are the main types of deep learning models and their general applications?",
    "What are the key advantages and challenges of deep learning?",
]
</EXAMPLE>

Provide your response in a python code list of str format:
"""

RERANK_PROMPT = """Based on the query questions and the retrieved chunk, to determine whether the chunk is helpful in answering any of the query question, you can only return "YES" or "NO", without any other information.

Query Questions: {query}
Retrieved Chunk: {retrieved_chunk}

Is the chunk helpful in answering the any of the questions?
"""


REFLECT_PROMPT = """Determine whether additional search queries are needed based on the original query, previous sub queries, and all retrieved document chunks. If further research is required, provide a Python list of up to 3 search queries. If no further research is required, return an empty list.

If the original query is to write a report, then you prefer to generate some further queries, instead return an empty list.

Original Query: {question}

Previous Sub Queries: {mini_questions}

Related Chunks: 
{mini_chunk_str}

Respond exclusively in valid List of str format without any other text."""


SUMMARY_PROMPT = """You are an expert academic writer tasked with creating a sophisticated synthesis based on the provided content architecture and retrieved information.

Content Architecture Plan:{architecture_plan}

Writing Context:
- Original Query: {question}
- Sub-queries Explored: {mini_questions}
- Content Type: {content_type}
- Structure Pattern: {structure_pattern}

Quality Standards for This Content:
- Analysis Depth: {analysis_depth}
- Evidence Requirements: {evidence_requirements}
- Critical Thinking: {critical_thinking}

Source Materials:
{mini_chunk_str}

Detailed Writing Instructions:
1. Role-Specific Expertise: 
Write as a leading expert in the relevant field with deep knowledge and analytical capabilities.

2. Structure Implementation:
    - Follow the provided outline structure precisely
   - Ensure logical flow between sections as planned
   - Implement the specified evidence allocation strategy

3. Analysis Requirements:
   - Go beyond information summarization to provide deep analytical insights  
   - Identify patterns, relationships, and implications not explicitly stated in sources
   - Present critical evaluation of different perspectives where applicable
   - Draw meaningful conclusions based on evidence synthesis

4.Evidence Integration：
   - Seamlessly weave evidence into argumentative flow
   - Prioritize high-quality, credible sources
   - Address potential counterarguments or limitations
   - Maintain clear traceability between claims and evidence
   
5.Language and Style Excellence：
   - Use precise, field-appropriate terminology consistently
   - Employ sophisticated academic discourse
   - Create smooth transitions that guide reader understanding
   - Vary sentence structure for engaging readability
   
6.Critical Thinking Demonstration：
   - Question assumptions and examine multiple perspectives
   - Identify cause-effect relationships and their implications
   - Highlight areas of uncertainty or ongoing debate
   - Suggest future research directions or practical applications

Output Format Requirements：

\\documentclass {{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{booktabs}}
\\usepackage{{parskip}}
\\setlength{{\\parindent}}{{0pt}}
\\setlength{{\\parskip}}{{1em}}

\\begin{{document}}

\\title{{[Generate an insightful title that captures the analytical focus]}}
\\author{{AI Research Synthesis}}
\\date{{\\today}}
\\maketitle

[Begin with your sophisticated academic analysis following the architecture plan...]

\\end{{document}}

Remember: 
This should read as an authoritative, insightful analysis that demonstrates expertise beyond simple information compilation. Focus on providing value through analytical depth and critical evaluation.
"""

QUERY_TYPE_ANALYSIS_PROMPT = """Analyze the following query to determine its type and complexity characteristics:

Query: {query}

Classify this query across the following dimensions:

1. Primary Intent:
   - Information seeking (what/who/when/where)
   - Analysis seeking (why/how/implications)
   - Comparison seeking (differences/similarities)
   - Synthesis seeking (comprehensive overview)
   - Problem-solving seeking (solutions/recommendations)

2. Domain Complexity:
   - Single domain focus
   - Multi-domain integration required
   - Cross-disciplinary analysis needed

3. Temporal Dimension:
   - Historical focus
   - Current state analysis
   - Future-oriented/predictive
   - Longitudinal development

4. Cognitive Demand:
   - Factual recall
   - Conceptual understanding
   - Analytical reasoning
   - Creative synthesis
   - Critical evaluation

Return as JSON:
{{
    "primary_intent": "string",
    "domain_complexity": "string", 
    "temporal_dimension": "string",
    "cognitive_demand": "string",
    "complexity_score": 1-10,
    "recommended_approach": "string"
}}
"""

BATCH_SUMMARY_PROMPT = """As a domain expert and information synthesizer, extract and synthesize the most relevant information from the following document chunks to answer the query with professional depth.

Query: {query}
Sub-queries: {sub_queries}
Content Type: {content_type}
Target Analysis Depth: {analysis_depth}

Document Chunks:
{chunk_content}

Provide a comprehensive synthesis that:
1. Identifies key facts, data points, and authoritative statements
2. Extracts domain-specific concepts and technical details
3. Recognizes patterns, trends, and relationships across sources
4. Notes methodological approaches and evidence quality
5. Identifies potential contradictions or knowledge gaps
6. Highlights unique insights or novel perspectives
7. Maintains source attribution for critical claims

Structure your synthesis with:
- Core Findings: Primary facts and data
- Technical Details: Methodologies, processes, specifications
- Analytical Insights: Patterns, implications, relationships
- Quality Assessment: Source reliability and evidence strength
- Knowledge Gaps: Missing information or uncertainties

Synthesis:"""

INTERMEDIATE_INTEGRATION_PROMPT = """As an expert synthesizer, integrate the following batch summaries into a coherent intermediate analysis.

Original Query: {query}
Content Architecture: {architecture_plan}

Batch Summaries:
{batch_summaries}

Create an integrated analysis that:
1. Identifies common themes and patterns across summaries
2. Resolves any contradictions or inconsistencies
3. Establishes logical connections between different information sources
4. Prepares a structured foundation for the final answer

Integrated Analysis:"""

FINAL_SYNTHESIS_PROMPT = """You are a leading expert researcher and academic writer tasked with creating a comprehensive, professional-quality report based on the integrated analysis.

Writing Context:
- Original Query: {query}
- Content Type: {content_type}
- Structure Pattern: {structure_pattern}
- Target Audience: {target_audience}
- Analysis Depth: {analysis_depth}
- Technical Level: {technical_level}

Quality Standards:
- Evidence Requirements: {evidence_requirements}
- Critical Thinking Level: {critical_thinking}
- Objectivity Level: {objectivity_level}

Content Architecture:
{content_outline}

Integrated Analysis:
{integrated_content}

Visualization Suggestions: {visualization_suggestions}

Professional Writing Requirements:
1. Expert Authority: Write with deep domain knowledge and analytical sophistication
2. Structured Argumentation: Present clear thesis, evidence, analysis, and conclusions
3. Critical Evaluation: Include multiple perspectives, limitations, and counterarguments
4. Evidence Integration: Seamlessly weave sources into coherent narrative
5. Professional Terminology: Use precise, field-appropriate language consistently
6. Logical Flow: Ensure smooth transitions and coherent progression
7. Analytical Depth: Go beyond summary to provide insights and implications

LaTeX Formatting Requirements:
- Use appropriate document structure with sections and subsections
- Include tables for comparative data or structured information
- Use professional academic formatting
- Include proper spacing and typography
- Add footnotes for additional context where appropriate
- Ensure mathematical expressions use proper LaTeX syntax if needed

Generate a sophisticated, publication-quality report in LaTeX format that demonstrates expertise beyond simple information compilation:

\\documentclass[12pt,article]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{parskip}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage{{fancyhdr}}

\\geometry{{margin=1in}}
\\setlength{{\\parindent}}{{0pt}}
\\setlength{{\\parskip}}{{1em}}

\\pagestyle{{fancy}}
\\fancyhf{{}}
\\rhead{{\\thepage}}
\\lhead{{Research Report}}

\\begin{{document}}

\\title{{[Generate a precise, professional title that captures the analytical focus and scope]}}
\\author{{AI Research Synthesis}}
\\date{{\\today}}
\\maketitle

\\tableofcontents
\\newpage

[Generate your comprehensive professional analysis following the content architecture...]

\\end{{document}}

Remember: This should read as an authoritative, insightful analysis that demonstrates deep expertise and critical thinking, not merely an information summary."""

# Token estimation constants
AVERAGE_TOKENS_PER_CHAR = 0.25
MAX_TOKENS_PER_REQUEST = 7500  # Leave buffer for model limits
BATCH_SIZE_TOKENS = 4000
PROMPT_OVERHEAD_TOKENS = 1500


class TokenManager:
    """Utility class for managing token counts and content optimization"""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for given text"""
        return int(len(text) * AVERAGE_TOKENS_PER_CHAR)

    @staticmethod
    def truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to approximate token limit while preserving sentence boundaries"""
        if TokenManager.estimate_tokens(text) <= max_tokens:
            return text

        # Estimate character limit
        char_limit = int(max_tokens / AVERAGE_TOKENS_PER_CHAR)

        if len(text) <= char_limit:
            return text

        # Find last complete sentence within limit
        truncated = text[:char_limit]
        last_sentence_end = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )

        if last_sentence_end > char_limit * 0.8:  # If we found a good sentence boundary
            return truncated[:last_sentence_end + 1]
        else:
            return truncated + "..."

    @staticmethod
    def create_batches(chunks: List[RetrievalResult], max_tokens_per_batch: int) -> List[List[RetrievalResult]]:
        """Create batches of chunks based on token limits"""
        batches = []
        current_batch = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = TokenManager.estimate_tokens(chunk.text)

            # If adding this chunk would exceed limit, start new batch
            if current_tokens + chunk_tokens > max_tokens_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = [chunk]
                current_tokens = chunk_tokens
            else:
                current_batch.append(chunk)
                current_tokens += chunk_tokens

        # Add remaining chunks
        if current_batch:
            batches.append(current_batch)

        return batches


@describe_class(
    "Enhanced DeepSearch agent with hierarchical processing for handling large-scale information retrieval "
    "while respecting token limits. Suitable for complex queries requiring comprehensive analysis."
)
class DeepSearch(RAGAgent):
    """
    Enhanced DeepSearch agent with hierarchical processing capabilities.

    Implements a multi-layer processing strategy to handle large amounts of retrieved content
    while staying within token limits through intelligent batching and progressive summarization.
    """

    def __init__(
            self,
            llm: BaseLLM,
            embedding_model: BaseEmbedding,
            vector_db: BaseVectorDB,
            max_iter: int = 3,
            route_collection: bool = True,
            text_window_splitter: bool = True,
            max_tokens_per_request: int = MAX_TOKENS_PER_REQUEST,
            **kwargs,
    ):
        """
        Initialize the enhanced DeepSearch agent.

        Args:
            llm: The language model to use for generating answers
            embedding_model: The embedding model for query embedding
            vector_db: The vector database for document search
            max_iter: Maximum iterations for search process
            route_collection: Whether to use collection routing
            text_window_splitter: Whether to use text window splitting
            max_tokens_per_request: Maximum tokens per LLM request
            **kwargs: Additional keyword arguments
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.max_iter = max_iter
        self.route_collection = route_collection
        self.collection_router = CollectionRouter(
            llm=self.llm, vector_db=self.vector_db, dim=embedding_model.dimension
        )
        self.text_window_splitter = text_window_splitter
        self.max_tokens_per_request = max_tokens_per_request
        self.token_manager = TokenManager()
        self.report_enhancer = ReportQualityEnhancer.ReportQualityEnhancer(
            self.llm
        )

    def _determine_processing_strategy(self, chunks: List[RetrievalResult]) -> str:
        """Determine the optimal processing strategy based on content volume"""
        total_tokens = sum(self.token_manager.estimate_tokens(chunk.text) for chunk in chunks)

        if total_tokens <= BATCH_SIZE_TOKENS:
            return "direct"
        elif total_tokens <= BATCH_SIZE_TOKENS * 5:
            return "two_layer"
        else:
            return "three_layer"

    def _process_batch(self, batch: List[RetrievalResult], query: str, sub_queries: List[str],
                               architecture_plan: Dict[str, Any]) -> Tuple[str, int]:
        """Process a single batch of chunks to generate batch summary"""
        # Prepare chunk content with token management
        chunk_texts = []
        available_tokens = self.max_tokens_per_request - PROMPT_OVERHEAD_TOKENS

        for chunk in batch:
            if self.text_window_splitter and "wider_text" in chunk.metadata:
                text = chunk.metadata["wider_text"]
            else:
                text = chunk.text
            chunk_texts.append(text)

        # Format and potentially truncate content
        formatted_content = self._format_chunk_texts_optimized(chunk_texts, available_tokens)

        # Generate batch summary
        batch_prompt = BATCH_SUMMARY_PROMPT.format(
            query=query,
            sub_queries=sub_queries,
            content_type=architecture_plan.get("content_type", "analysis"),
            analysis_depth=architecture_plan.get("quality_criteria", {}).get("analysis_depth", "comprehensive"),
            chunk_content=formatted_content
        )

        chat_response = self.llm.chat([{"role": "user", "content": batch_prompt}])
        return chat_response.content, chat_response.total_tokens

    def _integrate_batch_summaries(self, batch_summaries: List[str], query: str,
                                   architecture_plan: Dict[str, Any]) -> Tuple[str, int]:
        """Integrate multiple batch summaries into coherent intermediate analysis"""
        # Combine summaries with token management
        available_tokens = self.max_tokens_per_request - PROMPT_OVERHEAD_TOKENS
        combined_summaries = "\n\n".join([f"Summary {i + 1}:\n{summary}"
                                          for i, summary in enumerate(batch_summaries)])

        # Truncate if necessary
        combined_summaries = self.token_manager.truncate_to_tokens(
            combined_summaries, available_tokens
        )

        integration_prompt = INTERMEDIATE_INTEGRATION_PROMPT.format(
            query=query,
            architecture_plan=str(architecture_plan),
            batch_summaries=combined_summaries
        )

        chat_response = self.llm.chat([{"role": "user", "content": integration_prompt}])
        return chat_response.content, chat_response.total_tokens

    def _generate_final_answer(self, query: str, integrated_content: str,
                            architecture_plan: Dict[str, Any]) -> Tuple[str, int]:
        """Generate final comprehensive answer from integrated content"""
        # Prepare content outline
        content_outline = self._format_content_outline(architecture_plan.get("outline", []))

        # Ensure content fits within token limits
        available_tokens = self.max_tokens_per_request - PROMPT_OVERHEAD_TOKENS
        integrated_content = self.token_manager.truncate_to_tokens(
            integrated_content, available_tokens // 2
        )

        final_prompt = FINAL_SYNTHESIS_PROMPT.format(
            query=query,
            content_type=architecture_plan.get("content_type", "analysis"),
            structure_pattern=architecture_plan.get("structure_pattern", "hierarchical"),
            target_audience=architecture_plan.get("target_audience", "professional"),
            analysis_depth=architecture_plan.get("quality_criteria", {}).get("analysis_depth", "comprehensive"),
            technical_level=architecture_plan.get("quality_criteria", {}).get("technical_level", "advanced"),
            evidence_requirements=architecture_plan.get("quality_criteria", {}).get("evidence_requirements",
                                                                                    "multiple sources"),
            critical_thinking=architecture_plan.get("quality_criteria", {}).get("critical_thinking", "high"),
            objectivity_level=architecture_plan.get("quality_criteria", {}).get("objectivity_level", "balanced"),
            content_outline=content_outline,
            integrated_content=integrated_content,
            visualization_suggestions=str(architecture_plan.get("visualization_suggestions", []))
        )

        chat_response = self.llm.chat([{"role": "user", "content": final_prompt}])
        return chat_response.content, chat_response.total_tokens

    def _format_content_outline(self, outline: List[Dict[str, Any]]) -> str:
        """Format content outline for final synthesis"""
        if not outline:
            return "1. Introduction\n2. Main Analysis\n3. Conclusion"

        formatted_outline = []
        for i, section in enumerate(outline, 1):
            section_title = section.get("section", f"Section {i}")
            subsections = section.get("subsections", [])
            key_arguments = section.get("key_arguments", [])
            critical_points = section.get("critical_points", [])

            formatted_outline.append(f"{i}. {section_title}")

            if subsections:
                for j, subsection in enumerate(subsections[:3], 1):
                    formatted_outline.append(f"   {i}.{j} {subsection}")

            if key_arguments:
                formatted_outline.append("   Key Arguments:")
                for arg in key_arguments[:3]:
                    formatted_outline.append(f"   - {arg}")

            if critical_points:
                formatted_outline.append("   Critical Analysis Points:")
                for point in critical_points[:2]:
                    formatted_outline.append(f"   - {point}")

        return "\\n".join(formatted_outline)

    def _format_chunk_texts_optimized(self, chunk_texts: List[str], max_tokens: int) -> str:
        """Format chunk texts with intelligent token management"""
        if not chunk_texts:
            return "No content available."

        # Estimate tokens per chunk and adjust if needed
        total_estimated_tokens = sum(self.token_manager.estimate_tokens(text) for text in chunk_texts)

        if total_estimated_tokens <= max_tokens:
            # Can include all chunks
            return "\n".join([f"<chunk_{i}>\n{text}\n</chunk_{i}>"
                              for i, text in enumerate(chunk_texts)])
        else:
            # Need to truncate or limit chunks
            formatted_chunks = []
            used_tokens = 0

            for i, text in enumerate(chunk_texts):
                chunk_tokens = self.token_manager.estimate_tokens(text)

                if used_tokens + chunk_tokens > max_tokens:
                    # Truncate this chunk to fit remaining space
                    remaining_tokens = max_tokens - used_tokens
                    if remaining_tokens > 100:  # Only include if meaningful space left
                        truncated_text = self.token_manager.truncate_to_tokens(text, remaining_tokens)
                        formatted_chunks.append(f"<chunk_{i}>\n{truncated_text}\n</chunk_{i}>")
                    break
                else:
                    formatted_chunks.append(f"<chunk_{i}>\n{text}\n</chunk_{i}>")
                    used_tokens += chunk_tokens

            return "\n".join(formatted_chunks)

    def _hierarchical_processing(self, query: str, sub_queries: List[str],
                                        chunks: List[RetrievalResult],
                                        architecture_plan: Dict[str, Any]) -> Tuple[str, int]:
        """Main hierarchical processing logic"""
        strategy = self._determine_processing_strategy(chunks)
        total_tokens = 0

        log.color_print(f"<think> Processing strategy: {strategy} for {len(chunks)} chunks </think>\n")

        if strategy == "direct":
            # Direct processing for small amounts of content
            chunk_texts = []
            for chunk in chunks:
                if self.text_window_splitter and "wider_text" in chunk.metadata:
                    chunk_texts.append(chunk.metadata["wider_text"])
                else:
                    chunk_texts.append(chunk.text)

            available_tokens = self.max_tokens_per_request - PROMPT_OVERHEAD_TOKENS
            formatted_content = self._format_chunk_texts_optimized(chunk_texts, available_tokens)

            # Use simplified final synthesis for direct processing
            answer, tokens = self._generate_final_answer(query, formatted_content, architecture_plan)
            return answer, tokens

        elif strategy == "two_layer":
            # Two-layer processing: batches -> final answer
            batches = self.token_manager.create_batches(chunks, BATCH_SIZE_TOKENS)
            log.color_print(f"<think> Created {len(batches)} batches for two-layer processing </think>\n")

            # Process each batch
            batch_summaries = []
            for i, batch in enumerate(batches):
                log.color_print(f"<think> Processing batch {i + 1}/{len(batches)} </think>\n")
                summary, tokens = self._process_batch(batch, query, sub_queries, architecture_plan)
                batch_summaries.append(summary)
                total_tokens += tokens

            # Integrate summaries and generate final answer
            integrated_content = "\n\n".join([f"Batch {i + 1} Summary:\n{summary}"
                                              for i, summary in enumerate(batch_summaries)])

            answer, tokens = self._generate_final_answer(query, integrated_content, architecture_plan)
            total_tokens += tokens

            return answer, total_tokens

        else:  # three_layer
            # Three-layer processing: batches -> integration -> final answer
            batches = self.token_manager.create_batches(chunks, BATCH_SIZE_TOKENS)
            log.color_print(f"<think> Created {len(batches)} batches for three-layer processing </think>\n")

            # Layer 1: Process each batch
            batch_summaries = []
            for i, batch in enumerate(batches):
                log.color_print(f"<think> Processing batch {i + 1}/{len(batches)} </think>\n")
                summary, tokens = self._process_batch(batch, query, sub_queries, architecture_plan)
                batch_summaries.append(summary)
                total_tokens += tokens

            # Layer 2: Integrate batch summaries
            log.color_print("<think> Integrating batch summaries... </think>\n")
            integrated_content, tokens = self._integrate_batch_summaries(
                batch_summaries, query, architecture_plan
            )
            total_tokens += tokens

            # Layer 3: Generate final answer
            log.color_print("<think> Generating final comprehensive answer... </think>\n")
            answer, tokens = self._generate_final_answer(query, integrated_content, architecture_plan)
            total_tokens += tokens

            return answer, total_tokens

    # Keep existing methods from original implementation
    def _analyze_query_characteristics(self, query: str) -> Tuple[dict, int]:
        """Analyze query characteristics to provide a basis for content architecture planning"""
        chat_response = self.llm.chat(
            messages=[{"role": "user", "content": QUERY_TYPE_ANALYSIS_PROMPT.format(query=query)}]
        )
        response_content = chat_response.content
        return self.llm.literal_eval(response_content), chat_response.total_tokens

    def _plan_content_architecture(self, query: str, sub_queries: List[str],
                                          chunks: List[RetrievalResult],
                                          query_characteristics: Dict[str, Any]) -> Tuple[dict, int]:
        """Intelligent content architecture planning"""
        info_summary = self._generate_info_summary([chunk.text for chunk in chunks[:10]])

        content_architecture_prompt = CONTENT_ARCHITECTURE_PROMPT.format(
            query=query,
            sub_queries=sub_queries,
            info_summary=info_summary,
            query_characteristics=str(query_characteristics)
        )

        chat_response = self.llm.chat(
            messages=[{"role": "user", "content": content_architecture_prompt}],
            response_format={"type": "json_object"}
        )

        try:
            response_content = chat_response.content
            return self.llm.literal_eval(response_content), chat_response.total_tokens
        except:
            return {
                "content_type": "analysis",
                "structure_pattern": "hierarchical",
                "target_audience": "professional",
                "outline": [{"section": "Analysis", "key_arguments": [], "evidence_strategy": "comprehensive"}],
                "quality_criteria": {"analysis_depth": "comprehensive", "evidence_requirements": "multiple sources"},
                "visualization_suggestions": []
            }, chat_response.total_tokens

    def _generate_info_summary(self, chunk_texts: List[str]) -> str:
        """Generate information summaries for architecture planning"""
        if not chunk_texts:
            return "No information available."

        sample_texts = chunk_texts[:5]
        combined_text = " ".join(sample_texts)

        if len(combined_text) > 1000:
            combined_text = combined_text[:1000] + "..."

        return f"Available information covers: {combined_text}"

    def _generate_sub_queries(self, original_query: str) -> Tuple[List[str], int]:
        chat_response = self.llm.chat(
            messages=[
                {"role": "user", "content": SUB_QUERY_PROMPT.format(original_query=original_query)}
            ]
        )
        response_content = chat_response.content
        return self.llm.literal_eval(response_content), chat_response.total_tokens

    async def _search_chunks_from_vectordb(self, query: str, sub_queries: List[str]):
        # Implementation remains the same as original
        consume_tokens = 0
        if self.route_collection:
            selected_collections, n_token_route = self.collection_router.invoke(
                query=query, dim=self.embedding_model.dimension
            )
        else:
            selected_collections = self.collection_router.all_collections
            n_token_route = 0
        consume_tokens += n_token_route

        all_retrieved_results = []
        if not query.strip():
            log.color_print(f"<search> Skipping search for empty or whitespace query. </search>\n")
            return all_retrieved_results, consume_tokens

        log.color_print(f"<local_search> Searching locally for: '{query}' </local_search>\n")

        query_vector = self.embedding_model.embed_query(query)
        total_accepted_chunks = 0
        all_references = set()

        for collection in selected_collections:
            retrieved_results = self.vector_db.search_data(
                collection=collection, vector=query_vector
            )
            if not retrieved_results or len(retrieved_results) == 0:
                continue
            accepted_chunk_num = 0
            references = set()
            for retrieved_result in retrieved_results:
                rerank_queries = [q for q in ([query] + sub_queries) if q and q.strip()]
                if not rerank_queries:
                    all_retrieved_results.append(retrieved_result)
                    accepted_chunk_num += 1
                    references.add(retrieved_result.reference)
                    continue

                chat_response = self.llm.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": RERANK_PROMPT.format(
                                query=[query] + sub_queries,
                                retrieved_chunk=f"<chunk>{retrieved_result.text}</chunk>",
                            ),
                        }
                    ]
                )
                consume_tokens += chat_response.total_tokens
                response_content = chat_response.content.strip()
                if "<think>" in response_content and "</think>" in response_content:
                    end_of_think = response_content.find("</think>") + len("</think>")
                    response_content = response_content[end_of_think:].strip()
                if "YES" in response_content and "NO" not in response_content:
                    all_retrieved_results.append(retrieved_result)
                    accepted_chunk_num += 1
                    references.add(retrieved_result.reference)

            total_accepted_chunks += accepted_chunk_num
            all_references.update(references)

        if total_accepted_chunks > 0:
            log.color_print(
                f"<local_search> Found {total_accepted_chunks} relevant chunks from {len(all_references)} sources </local_search>\n"
            )
        else:
            log.color_print(
                f"<local_search> No relevant chunks found </local_search>\n"
            )

        return all_retrieved_results, consume_tokens

    def _generate_gap_queries(self, original_query: str, all_sub_queries: List[str],
                              all_chunks: List[RetrievalResult]) -> Tuple[List[str], int]:
        reflect_prompt = REFLECT_PROMPT.format(
            question=original_query,
            mini_questions=all_sub_queries,
            mini_chunk_str=self._format_chunk_texts_optimized([chunk.text for chunk in all_chunks], 2000)
            if len(all_chunks) > 0
            else "NO RELATED CHUNKS FOUND.",
        )
        chat_response = self.llm.chat([{"role": "user", "content": reflect_prompt}])
        response_content = chat_response.content
        return self.llm.literal_eval(response_content), chat_response.total_tokens

    def _calculate_information_gain(self, previous_results: List[RetrievalResult],
                                    new_results: List[RetrievalResult]) -> float:
        """Calculate information gain between iterations"""
        if not previous_results:
            return 1.0
        if not new_results:
            return 0.0

        prev_text = " ".join([r.text for r in previous_results])
        prev_embedding = self.embedding_model.embed_query(prev_text)

        new_text = " ".join([r.text for r in new_results])
        new_embedding = self.embedding_model.embed_query(new_text)

        similarity = np.dot(prev_embedding, new_embedding)
        information_gain = 1.0 - similarity
        volume_factor = min(len(new_results) / max(len(previous_results), 1), 1.0)

        return information_gain * volume_factor

    def retrieve(self, original_query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """Retrieve relevant documents from the knowledge base"""
        return asyncio.run(self.async_retrieve(original_query, **kwargs))

    async def async_retrieve(self, original_query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """Async retrieve implementation - same as original but with optimizations"""
        max_iter = kwargs.pop("max_iter", self.max_iter)
        min_info_gain_threshold = kwargs.pop("min_info_gain", 0.1)

        log.color_print(f"<query> {original_query} </query>\n")
        all_search_res = []
        all_sub_queries = []
        total_tokens = 0

        sub_queries, used_token = self._generate_sub_queries(original_query)
        total_tokens += used_token

        if not sub_queries:
            log.color_print("No sub queries were generated by the LLM. Using original query for search.")
            if original_query and original_query.strip():
                sub_queries = [original_query]
            else:
                log.color_print("Original query is also empty. Exiting retrieval.")
                return all_search_res, total_tokens, {"all_sub_queries": [], "iterations_performed": 0}
        else:
            log.color_print(f"<think> Break down the original query into new sub queries: {sub_queries}</think>\n")

        all_sub_queries.extend(sub_queries)
        sub_gap_queries = [q for q in sub_queries if q and q.strip()]

        previous_iteration_results = []
        iterations_performed = 0

        for iter in range(max_iter):
            iterations_performed = iter + 1
            log.color_print(f">> Iteration: {iterations_performed}\n")
            current_iteration_results = []

            if not sub_gap_queries:
                log.color_print(
                    f"<think> No valid sub-gap queries for iteration {iterations_performed}. Skipping search phase. </think>\n")
            else:
                search_tasks = [
                    self._search_chunks_from_vectordb(query, sub_gap_queries)
                    for query in sub_gap_queries
                ]
                search_results = await asyncio.gather(*search_tasks)

                for result in search_results:
                    search_res, consumed_token = result
                    total_tokens += consumed_token
                    current_iteration_results.extend(search_res)

            current_iteration_results = deduplicate_results(current_iteration_results)

            info_gain = self._calculate_information_gain(
                previous_iteration_results if previous_iteration_results else all_search_res,
                current_iteration_results
            )

            log.color_print(f"<think> Information gain from iteration {iter + 1}: {info_gain:.4f} </think>\n")

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
                log.color_print(f"<think> New search queries for next iteration: {sub_gap_queries} </think>\n")
                all_sub_queries.extend(sub_gap_queries)
                all_sub_queries = list(dict.fromkeys(all_sub_queries))

        additional_info = {
            "all_sub_queries": all_sub_queries,
            "iterations_performed": iterations_performed,
        }

        return all_search_res, total_tokens, additional_info

    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        Enhanced query method with hierarchical processing to handle token limits.

        Args:
            query (str): The query to answer
            **kwargs: Additional keyword arguments

        Returns:
            Tuple[str, List[RetrievalResult], int]: Answer, retrieved results, and token count
        """
        # Step 1: Retrieve relevant documents
        all_retrieved_results, n_token_retrieval, additional_info = self.retrieve(query, **kwargs)

        if not all_retrieved_results or len(all_retrieved_results) == 0:
            return f"No relevant information found for query '{query}'.", [], n_token_retrieval

        all_sub_queries = additional_info["all_sub_queries"]
        total_tokens = n_token_retrieval

        # Step 2: Analyze query characteristics
        query_characteristics, tokens_analysis = self._analyze_query_characteristics(query)
        total_tokens += tokens_analysis
        log.color_print(f"<think> Query characteristics: {query_characteristics} </think>\n")

        # Step 3: Plan content architecture
        architecture_plan, tokens_architecture = self._plan_content_architecture(
            query, all_sub_queries, all_retrieved_results, query_characteristics
        )
        total_tokens += tokens_architecture
        log.color_print(
            f"<think> Content architecture planned: {architecture_plan.get('structure_pattern', 'undefined')} </think>\n")

        # Step 4: Apply hierarchical processing to generate answer
        log.color_print(
            f"<think> Applying hierarchical processing to {len(all_retrieved_results)} retrieved chunks... </think>\n")

        answer, processing_tokens = self._hierarchical_processing(
            query, all_sub_queries, all_retrieved_results, architecture_plan
        )
        total_tokens += processing_tokens

        log.color_print("<think> Assessing and enhancing report quality... </think>\n")
        quality_assessment, quality_tokens = self.report_enhancer.assess_content_coherence(answer)
        total_tokens += quality_tokens

        log.color_print(f"<think> Report quality score: {quality_assessment.get('overall_score', 'N/A')} </think>\n")

        if quality_assessment.get("improvement_suggestions"):
            log.color_print(
                f"<think> Quality improvement suggestions: {quality_assessment['improvement_suggestions']} </think>\n")

        log.color_print("\n==== FINAL REPORT ====\n")
        log.color_print(answer)

        return answer, all_retrieved_results, total_tokens