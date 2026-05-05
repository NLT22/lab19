import json
import re
import time
from openai import OpenAI

SYSTEM_PROMPT = """You are a knowledge graph extraction assistant.
Extract all named entities and relationships from the given text.
Return ONLY a valid JSON array of triples, no explanation or markdown.
Each triple must have: {"subject": "...", "relation": "...", "object": "..."}
- subject and object are named entities (people, companies, products, places, technologies, dates)
- relation is a short verb phrase in UPPER_SNAKE_CASE (e.g. FOUNDED_BY, ACQUIRED, CEO_OF, LOCATED_IN)
- Normalize entity names to title case (e.g. "google" → "Google")
- Do not repeat identical triples"""

USER_TEMPLATE = "Text:\n{text}\n\nJSON triples:"


def _normalize(name: str) -> str:
    return " ".join(name.strip().title().split())


def _parse_triples(raw: str) -> list[tuple[str, str, str]]:
    """Parse JSON from LLM response, handling markdown code fences."""
    raw = raw.strip()
    # strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # try to extract first [...] block
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    triples = []
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        s = _normalize(str(item.get("subject", "")))
        r = str(item.get("relation", "")).strip().upper().replace(" ", "_")
        o = _normalize(str(item.get("object", "")))
        if s and r and o:
            key = (s, r, o)
            if key not in seen:
                seen.add(key)
                triples.append(key)
    return triples


class EntityRelationExtractor:
    def __init__(self, client: OpenAI, model: str, max_retries: int = 3):
        self.client = client
        self.model = model
        self.max_retries = max_retries

    def extract(self, text: str) -> list[tuple[str, str, str]]:
        """Extract (subject, relation, object) triples from text."""
        text = text[:4000]  # stay within context limits
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
                    ],
                    temperature=0,
                    max_tokens=2048,
                )
                raw = response.choices[0].message.content or ""
                triples = _parse_triples(raw)
                if triples:
                    return triples
            except Exception as e:
                print(f"  [extractor] attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        return []

    def batch_extract(self, chunks: list[dict], delay: float = 0.5) -> list[dict]:
        """Extract triples from a list of chunk dicts, return enriched list."""
        results = []
        for chunk in chunks:
            triples = self.extract(chunk["text"])
            results.append({**chunk, "triples": [list(t) for t in triples]})
            time.sleep(delay)
        return results
