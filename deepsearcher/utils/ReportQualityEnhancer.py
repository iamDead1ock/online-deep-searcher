from typing import List, Tuple, Dict

COHERENCE_PROMPT = """Analyze the following content for coherence, consistency, and professional quality.

Content:
{content}

Evaluate and score (0.0-1.0) the following aspects:
1. Logical flow and transitions
2. Argument consistency
3. Evidence integration quality
4. Technical accuracy
5. Professional tone
6. Structural organization

Provide specific improvement suggestions for scores below 0.8.

Return as JSON:
{{
    "scores": {{
        "logical_flow": 0.0-1.0,
        "argument_consistency": 0.0-1.0,
        "evidence_integration": 0.0-1.0,
        "technical_accuracy": 0.0-1.0,
        "professional_tone": 0.0-1.0,
        "structural_organization": 0.0-1.0
    }},
    "overall_score": 0.0-1.0,
    "improvement_suggestions": ["string"]
}}"""

SUMMARY_PROMPT = """Create a concise, professional executive summary for the following report.

Original Query: {query}
Full Report: {report}

Generate an executive summary that:
1. Captures the key findings and conclusions
2. Highlights the most significant insights
3. Provides clear takeaways for decision-makers
4. Maintains professional tone and clarity
5. Stays within 200-300 words

Executive Summary:"""

VIZ_PROMPT = """Analyze the content and suggest appropriate visualizations that would enhance understanding.

Content Type: {content_type}
Content: {content}

For each visualization suggestion, provide:
1. Visualization type (table, chart, diagram, etc.)
2. Specific data or concepts to visualize
3. Purpose and benefit
4. LaTeX implementation approach

Return as JSON array:
[
    {{
        "type": "string",
        "title": "string",
        "data_source": "string",
        "purpose": "string",
        "latex_approach": "string"
    }}
]"""

class ReportQualityEnhancer:
    """Enhanced report quality management with professional writing standards"""

    def __init__(self, llm):
        self.llm = llm

    def assess_content_coherence(self, content: str) -> Tuple[Dict[str, float], int]:
        """Assess content coherence and identify improvement areas"""
        coherence_prompt = COHERENCE_PROMPT.format(content=content[:2000])  # Limit for token management

        chat_response = self.llm.chat(
            messages = [{"role": "user", "content": coherence_prompt}],
            response_format={"type": "json_object"}
        )

        try:
            result = self.llm.literal_eval(chat_response.content)
            return result, chat_response.total_tokens
        except:
            return {"scores": {}, "overall_score": 0.7, "improvement_suggestions": []}, chat_response.total_tokens

    def generate_executive_summary(self, full_report: str, query: str) -> Tuple[str, int]:
        """Generate professional executive summary"""
        # Truncate report for token management
        truncated_report = full_report[:3000] if len(full_report) > 3000 else full_report
        summary_prompt = SUMMARY_PROMPT.format(query=query, report=truncated_report)

        chat_response = self.llm.chat(
            messages = [{"role": "user", "content": summary_prompt.format(query=query, report=truncated_report)}]
        )

        return chat_response.content, chat_response.total_tokens

    def suggest_visualizations(self, content: str, content_type: str) -> Tuple[List[Dict], int]:
        """Suggest appropriate visualizations for the content"""
        viz_prompt = VIZ_PROMPT.format(content_type=content_type, content=content[:2000])

        chat_response = self.llm.chat(
            messages = [{"role": "user", "content": viz_prompt.format(content_type=content_type, content=content[:2000])}],
            response_format={"type": "json_object"}
        )

        try:
            result = self.llm.literal_eval(chat_response.content)
            return result, chat_response.total_tokens
        except:
            return [], chat_response.total_tokens