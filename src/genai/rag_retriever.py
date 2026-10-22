# =========================================================
# SMART CITY RAG RETRIEVER
# =========================================================

from pathlib import Path
import re


# =========================================================
# PATH TO KNOWLEDGE BASE
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT
    / "src"
    / "genai"
    / "knowledge_base"
    / "smart_city_knowledge.txt"
)


# =========================================================
# LOAD KNOWLEDGE
# =========================================================

def load_knowledge():

    with open(
        KNOWLEDGE_BASE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# =========================================================
# SPLIT KNOWLEDGE INTO SECTIONS
# =========================================================

def create_sections(text):

    sections = {}

    # Split using headings written in uppercase
    parts = re.split(
        r"\n(?=[A-Z][A-Z\s\-]+\n)",
        text
    )

    for part in parts:

        lines = part.strip().splitlines()

        if not lines:
            continue

        title = lines[0].strip()

        content = "\n".join(
            lines[1:]
        ).strip()

        if content:
            sections[title] = content

    return sections


# =========================================================
# SIMPLE KEYWORD RETRIEVAL
# =========================================================

def retrieve(
    query,
    top_k=3
):

    knowledge = load_knowledge()

    sections = create_sections(
        knowledge
    )

    query_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            query.lower()
        )
    )

    results = []

    for title, content in sections.items():

        section_text = (
            title + " " + content
        ).lower()

        section_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                section_text
            )
        )

        matching_words = (
            query_words & section_words
        )

        score = len(
            matching_words
        )

        if score > 0:

            results.append(
                {
                    "title": title,
                    "content": content,
                    "score": score
                }
            )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# DISPLAY RETRIEVED INFORMATION
# =========================================================

def display_results(
    query,
    results
):

    print()
    print("=" * 70)
    print("RAG RETRIEVAL")
    print("=" * 70)

    print()
    print(
        f"Query: {query}"
    )

    print()

    if not results:

        print(
            "No relevant knowledge was found."
        )

        return

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"[{index}] "
            f"{result['title']}"
        )

        print(
            f"Relevance score: "
            f"{result['score']}"
        )

        print()

        print(
            result["content"]
        )

        print()
        print("-" * 70)


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    test_query = (
        "Why did the DQN recommend "
        "traffic reduction?"
    )

    results = retrieve(
        test_query,
        top_k=3
    )

    display_results(
        test_query,
        results
    )