import os
import json
import requests
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# ================== CONFIG ==================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Set it in terminal using:\n"
        '$env:GROQ_API_KEY="your_key_here"'
    )

client = Groq(api_key=api_key)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

valid_extensions = (".py", ".ipynb", ".md", ".js", ".java", ".cpp")
VECTOR_STORE = []


# ================== LLM ==================

def ask_llm(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert codebase analyst. Be accurate, concise, and do not guess."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"AI Error: {e}"


# ================== FILE HANDLING ==================

def extract_ipynb_code(raw):
    try:
        notebook = json.loads(raw)
        code_cells = []

        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                code_cells.append(source)

        return "\n\n".join(code_cells)

    except Exception:
        return raw


def is_valid_file(filename):
    ignored = (
        "license",
        "code_of_conduct",
        "contributing",
        "pull_request_template",
        "issue_template"
    )

    lower_name = filename.lower()

    if any(word in lower_name for word in ignored):
        return False

    return lower_name.endswith(valid_extensions)


def chunk_text(text, size=600, overlap=120):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap

    return chunks


# ================== EMBEDDINGS ==================

def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0

    return np.dot(a, b) / denominator


def build_vector_store(results):
    global VECTOR_STORE
    VECTOR_STORE = []

    for result in results:
        chunks = chunk_text(result["content"])

        if not chunks:
            continue

        embeddings = embedder.encode(chunks)

        for chunk, embedding in zip(chunks, embeddings):
            VECTOR_STORE.append({
                "file": result["file"],
                "chunk": chunk,
                "embedding": embedding
            })


def retrieve_relevant_chunks(question, top_k=5):
    if not VECTOR_STORE:
        return []

    question_lower = question.lower()
    question_embedding = embedder.encode([question])[0]

    scored_chunks = []

    for item in VECTOR_STORE:
        score = cosine_similarity(question_embedding, item["embedding"])

        text = item["chunk"].lower()
        file = item["file"].lower()

        # 🔥 Keyword boost
        keywords = ["model", "train", "predict", "fit", "classifier", "regression"]

        if any(k in text for k in keywords):
            score += 0.15

        if any(k in question_lower for k in keywords) and any(k in text for k in keywords):
            score += 0.25

        # File priority
        if file.endswith(".ipynb"):
            score += 0.1

        scored_chunks.append((score, item))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    return [item for score, item in scored_chunks[:top_k]]


# ================== AI FEATURES ==================

def explain_file(content, filename):
    prompt = f"""
Explain this file clearly and briefly.

File name: {filename}

Return in this format:

Purpose:
Key logic:
Technologies used:

File content:
{content[:1800]}
"""
    return ask_llm(prompt)


def get_summary(results):
    context = ""

    for result in results:
        context += f"\nFILE: {result['file']}\n{result['content'][:1000]}\n"

    prompt = f"""
Summarize this codebase in a short, useful format.

Return only:

One-line description:
Key features:
Tech stack:
Main purpose:

Codebase context:
{context[:4500]}
"""
    return ask_llm(prompt)


def ask_question(question):
    relevant_chunks = retrieve_relevant_chunks(question)

    if not relevant_chunks:
        return "Please analyze a repository first."

    context = ""

    for item in relevant_chunks:
        context += f"\nFILE: {item['file']}\n{item['chunk']}\n"

    prompt = f"""
You are an expert software engineer analyzing a codebase.

Answer the question using ONLY the provided context.

================ RULES ================

1. Grounded Answers
- Do NOT guess.
- If information is missing, say: "Not found in analyzed files".

2. Precision
- Identify exact names (functions, classes, models, APIs, variables).
- Quote code snippets when helpful.

3. Relevance
- Ignore unrelated content.
- Focus only on parts relevant to the question.

4. Role Understanding
- Identify the ROLE of each component:
  • Core logic (main functionality)
  • Supporting logic (helpers, preprocessing, utilities)
  • External tools (libraries, APIs)

5. Importance Ranking
- Prioritize components that directly answer the question.
- Supporting tools should be mentioned separately if needed.

6. File Awareness
- Mention file names where information is found.

7. Clarity
- Keep answer structured and concise.
- Use bullet points when needed.

Structure your answer as:

Primary Answer:
Supporting Details:
Not Relevant / Supporting Components:
=====================================

Context:
{context}

Question:
{question}

Answer:
"""
    return ask_llm(prompt)


# ================== GITHUB ==================

def analyze_repo(repo_url):
    parts = repo_url.strip().rstrip("/").split("/")

    if len(parts) < 5:
        raise ValueError("Invalid GitHub repo URL")

    owner = parts[-2]
    repo = parts[-1]

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"

    results = []
    state = {
        "count": 0,
        "limit": 8
    }

    def fetch_files(url):
        if state["count"] >= state["limit"]:
            return

        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"GitHub API Error: {response.status_code}")

        data = response.json()

        for item in data:
            if state["count"] >= state["limit"]:
                return

            name = item["name"]

            if item["type"] == "file" and item.get("download_url"):
                if not is_valid_file(name):
                    continue

                raw = requests.get(item["download_url"]).text

                if name.endswith(".ipynb"):
                    content = extract_ipynb_code(raw)
                else:
                    content = raw

                if not content.strip():
                    continue

                explanation = explain_file(content, name)

                results.append({
                    "file": name,
                    "content": content[:3000],
                    "explanation": explanation
                })

                state["count"] += 1

            elif item["type"] == "dir":
                fetch_files(item["url"])

    fetch_files(api_url)
    build_vector_store(results)

    return results