from typing import Dict


def generation_prompt(topic: str, audience: str, tone: str, format_name: str, length_hint: str) -> str:
    return f"""
You are a helpful writing assistant.

Task: Generate {format_name} content.

Topic:
{topic}

Audience:
{audience}

Tone:
{tone}

Length:
{length_hint}

Rules:
- Be specific and practical.
- Avoid fluff; include concrete examples where possible.
- If you need assumptions, state them clearly.
""".strip()


def summarization_prompt(text: str, style: str) -> str:
    return f"""
You are a precise summarizer.

Summarization style:
{style}

Input text:
{text}

Output EXACTLY in this markdown format:

## TL;DR
- ...

## Key Points
- ...

## Action Items
- ...

## Risks / Open Questions
- ...
""".strip()


def merge_prompt(chunk_summaries: str, style: str) -> str:
    return f"""
You are combining summaries of chunks into one consistent final summary.

Summarization style:
{style}

Chunk summaries:
{chunk_summaries}

Write the final summary in EXACTLY this markdown format:

## TL;DR
- ...

## Key Points
- ...

## Action Items
- ...

## Risks / Open Questions
- ...
""".strip()


def compare_judge_prompt(outputs: Dict[str, str], goal: str) -> str:
    joined = "\n\n".join([f"### {k}\n{v}" for k, v in outputs.items()])
    return f"""
You are judging multiple candidate outputs for the same user request.

User goal:
{goal}

Candidates:
{joined}

Pick the best candidate and explain why in 5-8 bullets.
Then provide a short improvement suggestion for each non-winning candidate.

Return markdown with:
## Winner
<model name>

## Why Winner
- ...

## Improvements
### <model name>
- ...
""".strip()
