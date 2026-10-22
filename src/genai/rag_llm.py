# =========================================================
# RAG + LOCAL LLM
# =========================================================

import sys
from pathlib import Path

import ollama


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GENAI_PATH = (
    PROJECT_ROOT
    / "src"
    / "genai"
)

sys.path.append(
    str(GENAI_PATH)
)


# =========================================================
# IMPORT RAG RETRIEVER
# =========================================================

from rag_retriever import retrieve


# =========================================================
# MODEL
# =========================================================

MODEL_NAME = "llama3.2:3b"


# =========================================================
# GENERATE RAG-BASED ANSWER
# =========================================================

def generate_answer(question):

    # -----------------------------------------------------
    # Retrieve relevant knowledge
    # -----------------------------------------------------

    results = retrieve(
        question,
        top_k=3
    )

    if not results:

        context = (
            "No relevant information was found "
            "in the project knowledge base."
        )

    else:

        context_parts = []

        for result in results:

            context_parts.append(
                f"SECTION: {result['title']}\n"
                f"{result['content']}"
            )

        context = "\n\n".join(
            context_parts
        )

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""
You are the AI explanation assistant for a
Smart City Digital Twin project.

Answer the user's question using ONLY the
provided project knowledge.

PROJECT KNOWLEDGE:
------------------
{context}
------------------

USER QUESTION:
{question}

Instructions:

1. Give a clear and concise answer.
2. Use the retrieved project knowledge.
3. Do not invent measurements or facts.
4. Clearly distinguish between real observations,
   machine-learning predictions, computer-vision
   estimates, simulations, and reinforcement-learning
   recommendations.
5. If the knowledge base does not contain enough
   information, say so.
6. Remember that the project's environmental impact
   values are simulation-based indicators and are
   not validated real-world AQI predictions.

Answer:
"""

    # -----------------------------------------------------
    # Call local LLM
    # -----------------------------------------------------

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    question = (
        "Why did the DQN recommend "
        "traffic reduction?"
    )

    print("=" * 70)
    print("SMART CITY RAG + LOCAL LLM")
    print("=" * 70)

    print()
    print(
        f"Question: {question}"
    )

    print()
    print("Generating answer...")
    print()

    answer = generate_answer(
        question
    )

    print("=" * 70)
    print("AI ANSWER")
    print("=" * 70)

    print()
    print(answer)

    print()
    print("=" * 70)
    print("RAG + LLM TEST COMPLETE")
    print("=" * 70)