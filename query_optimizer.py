import re


EXPANSIONS = {
    "ai": "artificial intelligence",
    "eu": "european union",
    "act": "regulation law act",
    "risk": "risk category classification",
}


def optimize_query(question: str) -> str:
    """Normalize and lightly expand the query without changing user intent."""
    clean = re.sub(r"\s+", " ", question.strip())
    if not clean:
        return clean

    terms = re.findall(r"[a-zA-Z0-9]+", clean.lower())
    additions = []
    for term in terms:
        if term in EXPANSIONS:
            additions.append(EXPANSIONS[term])

    if additions:
        return f"{clean} {' '.join(additions)}"
    return clean
