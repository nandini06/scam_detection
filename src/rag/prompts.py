from src.rag.retrieval import format_retrieved_context, summarize_retrieved_labels


def build_rag_prompt(
    query_text: str,
    retrieved_df,
) -> str:
    """
    Build RAG classification prompt.
    """

    retrieved_context = format_retrieved_context(retrieved_df)
    label_summary = summarize_retrieved_labels(retrieved_df)

    prompt = f"""
You are a scam detection assistant designed to help protect elderly users from fraud and manipulation.

Your task is to analyze the current conversation and classify it.

Allowed labels:
- SAFE: The conversation appears normal, legitimate, or low risk.
- RISKY: The conversation shows scam-like behavior, manipulation, urgency, impersonation, payment pressure, or requests for sensitive information.
- UNCERTAIN: There is not enough evidence yet to confidently classify the conversation.

Use the retrieved examples as supporting evidence, but do not blindly copy their labels.
Focus on the current conversation.

Retrieved label summary:
{label_summary}

Retrieved similar examples:
{retrieved_context}

Current conversation to classify:
{query_text}

Return your answer in this exact format:

Label: SAFE/RISKY/UNCERTAIN
Reason: brief explanation
Evidence: specific phrases or behaviors from the conversation
Recommended action: no intervention / continue monitoring / warn user / warn user and trusted contact
""".strip()

    return prompt