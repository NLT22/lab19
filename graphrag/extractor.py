import asyncio
import json
import re
import time
from openai import OpenAI, AsyncOpenAI
from graphrag.config import chat_kwargs

SYSTEM_PROMPT = """/no_think
You are a knowledge graph extraction assistant. Extract high-value facts from the text.
Return ONLY a valid JSON array. No markdown, no explanation.

Each element must use this schema:
{
  "subject": "...",
  "subject_type": "Company|Person|Product|Technology|Place|Organization|Other",
  "relation": "UPPER_SNAKE_CASE",
  "object": "...",
  "object_type": "Company|Person|Product|Technology|Place|Date|Money|Percent|Number|Text|Other",
  "evidence": "short exact phrase from the text supporting this fact",
  "confidence": 0.0-1.0
}

Extract BOTH entity-to-entity relationships and literal attributes such as money,
percentages, dates, employee counts, revenue, ownership shares, valuations, missions,
market share, products, services, and headquarters.

Preserve exact casing and formatting (OpenAI stays OpenAI, NVIDIA stays NVIDIA,
GPT-4 stays GPT-4, US$13.1 billion stays US$13.1 billion).

RELATIONS — pay close attention to subject/object direction:
- FOUNDED_BY:    Company --[FOUNDED_BY]--> Person      e.g. OpenAI --[FOUNDED_BY]--> Sam Altman
- FOUNDED_IN:    Company --[FOUNDED_IN]--> Year
- CEO_OF:        Person  --[CEO_OF]--> Company         e.g. Sam Altman --[CEO_OF]--> OpenAI
- INVESTED_IN:   Investor --[INVESTED_IN]--> Company   e.g. Microsoft --[INVESTED_IN]--> OpenAI
- ACQUIRED:      Acquirer --[ACQUIRED]--> Target        full buyout only, NOT investment/partnership
- RELEASED:      Company --[RELEASED]--> Product        e.g. Nvidia --[RELEASED]--> H100
- PROVIDES:      Company --[PROVIDES]--> Service        e.g. Microsoft --[PROVIDES]--> Azure OpenAI Service
- PARTNERED_WITH: Company --[PARTNERED_WITH]--> Company
- SUBSIDIARY_OF: Division --[SUBSIDIARY_OF]--> Parent   legal ownership only, NOT investment
- LOCATED_IN:    Entity --[LOCATED_IN]--> Place
- WORKS_FOR:     Person --[WORKS_FOR]--> Company
- HAS_MISSION:   Company --[HAS_MISSION]--> mission_description
- VALUED_AT:     Company --[VALUED_AT]--> Amount
- HAS_REVENUE:   Company --[HAS_REVENUE]--> Amount
- HAS_NET_INCOME: Company --[HAS_NET_INCOME]--> Amount
- HAS_EMPLOYEES: Company --[HAS_EMPLOYEES]--> Number
- HAS_OWNERSHIP_SHARE: Owner/Company --[HAS_OWNERSHIP_SHARE]--> Percent
- HAS_MARKET_SHARE: Company --[HAS_MARKET_SHARE]--> Percent
- RAISED:        Company --[RAISED]--> Amount           for funding rounds
- USES:          Entity --[USES]--> Technology
- OWNS:          Company --[OWNS]--> Asset/Platform

Use the most specific relation possible. Do not repeat identical facts."""

# Canonicalize common relation variants to a single preferred form
RELATION_CANONICAL: dict[str, str] = {
    # CEO / leadership
    "HAS_CEO": "CEO_OF", "LED_BY": "CEO_OF", "HEADED_BY": "CEO_OF",
    "IS_CEO_OF": "CEO_OF", "LEADS": "CEO_OF", "IS_PRESIDENT_OF": "CEO_OF",
    "CEO": "CEO_OF", "HAS_LEADER": "CEO_OF", "CHIEF_EXECUTIVE_OF": "CEO_OF",
    # Founding
    "CO_FOUNDED_BY": "FOUNDED_BY", "WAS_FOUNDED_BY": "FOUNDED_BY",
    "ESTABLISHED_BY": "FOUNDED_BY", "CREATED_BY": "FOUNDED_BY",
    "CO_FOUNDER_OF": "FOUNDED_BY", "FOUNDED": "FOUNDED_BY",
    "FOUNDERS_BY": "FOUNDED_BY",  # common LLM typo
    "FOUNDER_OF": "FOUNDED_BY",
    # Investment
    "INVESTED_INTO": "INVESTED_IN", "HAS_INVESTED_IN": "INVESTED_IN",
    "IS_INVESTOR_IN": "INVESTED_IN", "FUNDING_FROM": "INVESTED_IN",
    "RECEIVED_INVESTMENT_FROM": "INVESTED_IN", "BACKED_BY": "INVESTED_IN",
    "ATTRACTS_INVESTMENT_FROM": "INVESTED_IN",
    # Fundraising / valuation
    "RAISED_FUNDING": "RAISED", "HAS_VALUATION": "VALUED_AT",
    "IS_VALUED_AT": "VALUED_AT", "WORTH": "VALUED_AT",
    "REVENUE": "HAS_REVENUE", "HAS_TOTAL_REVENUE": "HAS_REVENUE",
    "REPORTS_REVENUE": "HAS_REVENUE", "GENERATED_REVENUE": "HAS_REVENUE",
    "NET_INCOME": "HAS_NET_INCOME", "HAS_LOSS": "HAS_NET_INCOME",
    "REPORTS_NET_INCOME": "HAS_NET_INCOME",
    "EMPLOYEES": "HAS_EMPLOYEES", "HAS_EMPLOYEE_COUNT": "HAS_EMPLOYEES",
    "NUMBER_OF_EMPLOYEES": "HAS_EMPLOYEES",
    "OWNERSHIP": "HAS_OWNERSHIP_SHARE", "OWNED_PERCENTAGE": "HAS_OWNERSHIP_SHARE",
    "HAS_OWNER": "HAS_OWNERSHIP_SHARE", "HAS_MARKET_SHARE_OF": "HAS_MARKET_SHARE",
    # Products / services
    "LAUNCHED": "RELEASED", "INTRODUCED": "RELEASED",
    "HAS_PRODUCT": "RELEASED", "MADE": "RELEASED", "DEVELOPED": "RELEASED",
    "OFFERS": "PROVIDES", "PROVIDES_SERVICE": "PROVIDES",
    # Location
    "BASED_IN": "LOCATED_IN", "HEADQUARTERED_IN": "LOCATED_IN",
    "HAS_HEADQUARTERS_IN": "LOCATED_IN", "LOCATED_AT": "LOCATED_IN",
    # Partnership
    "HAS_PARTNERSHIP_WITH": "PARTNERED_WITH", "IN_PARTNERSHIP_WITH": "PARTNERED_WITH",
    "COLLABORATED_WITH": "PARTNERED_WITH", "WORKS_WITH": "PARTNERED_WITH",
    "COLLABORATION_WITH": "PARTNERED_WITH", "IN_COLLABORATION_WITH": "PARTNERED_WITH",
    # Employment
    "WORKS_AT": "WORKS_FOR", "EMPLOYED_BY": "WORKS_FOR",
    "IS_EMPLOYEE_OF": "WORKS_FOR", "MEMBER_OF": "WORKS_FOR",
    # Acquisition / ownership
    "BOUGHT": "ACQUIRED", "PURCHASED": "ACQUIRED",
    "PART_OF": "SUBSIDIARY_OF", "OWNED_BY": "SUBSIDIARY_OF",
    "IS_DIVISION_OF": "SUBSIDIARY_OF",
    # Mission
    "MISSION": "HAS_MISSION", "HAS_GOAL": "HAS_MISSION",
}

USER_TEMPLATE = "Text:\n{text}\n\nJSON facts:"


ENTITY_TYPES = {
    "COMPANY", "PERSON", "PRODUCT", "TECHNOLOGY", "PLACE", "ORGANIZATION", "OTHER"
}

OBJECT_TYPES = ENTITY_TYPES | {"DATE", "MONEY", "PERCENT", "NUMBER", "TEXT"}

ALLOWED_RELATIONS = {
    "FOUNDED_BY", "FOUNDED_IN", "CEO_OF", "INVESTED_IN", "ACQUIRED",
    "RELEASED", "PROVIDES", "PARTNERED_WITH", "SUBSIDIARY_OF",
    "LOCATED_IN", "WORKS_FOR", "HAS_MISSION", "VALUED_AT",
    "HAS_REVENUE", "HAS_NET_INCOME", "HAS_EMPLOYEES",
    "HAS_OWNERSHIP_SHARE", "HAS_MARKET_SHARE", "RAISED", "USES", "OWNS",
}


def _normalize(name: str) -> str:
    """Capitalize first letter of all-lowercase words; preserve casing of mixed/upper words."""
    parts = name.strip().split()
    return " ".join(p.capitalize() if p.islower() else p for p in parts)


def _clean_type(value: str, allowed: set[str], default: str) -> str:
    value = str(value or "").strip().upper().replace(" ", "_")
    return value if value in allowed else default


def _parse_triples(raw: str) -> list[dict]:
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

    facts: list[dict] = []
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        subject_type = _clean_type(item.get("subject_type", "Other"), ENTITY_TYPES, "OTHER")
        object_type = _clean_type(item.get("object_type", "Other"), OBJECT_TYPES, "OTHER")
        s = _normalize(str(item.get("subject", "")))
        r = str(item.get("relation", "")).strip().upper().replace(" ", "_")
        r = RELATION_CANONICAL.get(r, r)
        if r not in ALLOWED_RELATIONS:
            continue
        o_raw = str(item.get("object", "")).strip()
        o = _normalize(o_raw) if object_type in ENTITY_TYPES else " ".join(o_raw.split())
        evidence = " ".join(str(item.get("evidence", "")).strip().split())[:500]
        try:
            confidence = float(item.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        confidence = max(0.0, min(1.0, confidence))
        if s and r and o:
            key = (s, r, o)
            if key not in seen:
                seen.add(key)
                facts.append({
                    "subject": s,
                    "subject_type": subject_type.title(),
                    "relation": r,
                    "object": o,
                    "object_type": object_type.title(),
                    "evidence": evidence,
                    "confidence": confidence,
                })
    return facts


class EntityRelationExtractor:
    def __init__(self, client: OpenAI, model: str, max_retries: int = 3):
        self.client = client
        self.model = model
        self.max_retries = max_retries

    def extract(self, text: str) -> list[dict]:
        """Extract structured facts from text."""
        text = text[:4000]  # stay within context limits
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
                    ],
                    max_completion_tokens=2048,
                    **chat_kwargs(temperature=0),
                )
                raw = response.choices[0].message.content or ""
                facts = _parse_triples(raw)
                if facts:
                    return facts
            except Exception as e:
                print(f"  [extractor] attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        return []

    def batch_extract(self, chunks: list[dict], delay: float = 0.5) -> list[dict]:
        """Extract triples from a list of chunk dicts, return enriched list."""
        results = []
        for chunk in chunks:
            triples = self.extract(chunk["text"])
            results.append({**chunk, "triples": triples})
            time.sleep(delay)
        return results


class AsyncEntityRelationExtractor:
    """Async extractor — use with asyncio for parallel chunk processing."""

    def __init__(self, client: AsyncOpenAI, model: str, max_retries: int = 3):
        self.client = client
        self.model = model
        self.max_retries = max_retries

    async def aextract(self, text: str, timeout: float = 120.0) -> list[dict]:
        """Async extract facts. Skips chunk and returns [] if timeout exceeded."""
        text = text[:4000]
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": USER_TEMPLATE.format(text=text)},
                        ],
                        max_completion_tokens=2048,
                        **chat_kwargs(temperature=0),
                    ),
                    timeout=timeout,
                )
                raw = response.choices[0].message.content or ""
                facts = _parse_triples(raw)
                if facts:
                    return facts
            except asyncio.TimeoutError:
                print(f"  [extractor] timeout after {timeout}s — skipping chunk")
                return []
            except Exception as e:
                wait = 2 ** attempt
                print(f"  [extractor] attempt {attempt + 1} failed: {e} (retry in {wait}s)")
                await asyncio.sleep(wait)
        return []
