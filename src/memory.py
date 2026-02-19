"""Selective memory: write high-signal facts to USER_MEMORY.md and COMPANY_MEMORY.md."""

import json
import os
from pathlib import Path

from src.llm_client import get_client, get_model


USER_MEMORY_PATH = Path("USER_MEMORY.md")
COMPANY_MEMORY_PATH = Path("COMPANY_MEMORY.md")

MEMORY_PROMPT = """Analyze this conversation turn. Extract ONLY high-signal, reusable facts worth remembering.
Rules:
- USER facts: personal preferences, role, workflow preferences (e.g., "User prefers weekly summaries on Mondays", "User is a Project Finance Analyst")
- COMPANY facts: org-wide learnings, workflows, bottlenecks (e.g., "Asset Management interfaces with Project Finance", "Recurring bottleneck is X")
- Do NOT store: raw transcript, secrets, PII, low-value chitchat
- Be selective: only 0-2 facts per turn, high confidence only

Conversation turn:
{turn}

Respond with a JSON array of objects. Each object: {{"target": "USER" or "COMPANY", "summary": "brief fact", "confidence": 0.0-1.0}}
If nothing worth storing, return: []
Example: [{{"target": "USER", "summary": "User prefers weekly summaries on Mondays.", "confidence": 0.9}}]"""


def extract_memory_candidates(
    user_message: str,
    assistant_message: str,
    api_key: str | None = None,
) -> list[dict]:
    """
    Use LLM to decide what (if anything) to store. Returns list of
    {target, summary, confidence}. Only include items with confidence >= 0.8.
    Works with Ollama (USE_OLLAMA=1) or OpenAI.
    """
    client = get_client(api_key=api_key)
    if not client:
        return []

    turn = f"User: {user_message}\nAssistant: {assistant_message}"
    prompt = MEMORY_PROMPT.format(turn=turn)

    try:
        response = client.chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content or "[]"
    except Exception:
        return []

    # Parse JSON (handle markdown code blocks)
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else "[]"

    try:
        items = json.loads(content)
    except json.JSONDecodeError:
        return []

    return [i for i in items if isinstance(i, dict) and i.get("confidence", 0) >= 0.8]


def append_to_memory(target: str, summary: str, base_path: Path) -> None:
    """Append one fact to USER_MEMORY.md or COMPANY_MEMORY.md."""
    if target == "USER":
        path = USER_MEMORY_PATH
    elif target == "COMPANY":
        path = COMPANY_MEMORY_PATH
    else:
        return

    line = f"- {summary}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def process_memory(
    user_message: str,
    assistant_message: str,
    base_path: Path | None = None,
    api_key: str | None = None,
) -> list[dict]:
    """
    Extract memory candidates, append to files, return list of {target, summary}
    for demo.memory_writes.
    """
    base_path = base_path or Path(".")
    candidates = extract_memory_candidates(user_message, assistant_message, api_key)

    # Avoid duplicates: read existing content and skip if summary already present
    existing_user = set()
    existing_company = set()
    if USER_MEMORY_PATH.exists():
        for line in USER_MEMORY_PATH.read_text().splitlines():
            if line.strip().startswith("-"):
                existing_user.add(line.strip()[1:].strip().lower())
    if COMPANY_MEMORY_PATH.exists():
        for line in COMPANY_MEMORY_PATH.read_text().splitlines():
            if line.strip().startswith("-"):
                existing_company.add(line.strip()[1:].strip().lower())

    written = []
    for c in candidates:
        target = c.get("target", "").upper()
        summary = (c.get("summary") or "").strip()
        if not summary or target not in ("USER", "COMPANY"):
            continue
        existing = existing_user if target == "USER" else existing_company
        if summary.lower() in existing:
            continue
        append_to_memory(target, summary, base_path)
        written.append({"target": target, "summary": summary})
        if target == "USER":
            existing_user.add(summary.lower())
        else:
            existing_company.add(summary.lower())

    return written


def load_memory_for_context(target: str) -> str:
    """Load USER or COMPANY memory as context string for the RAG prompt."""
    path = USER_MEMORY_PATH if target == "USER" else COMPANY_MEMORY_PATH
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()
