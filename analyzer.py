import requests
import json
import os
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# ================== SETUP ==================

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

valid_extensions = (".py", ".ipynb", ".md")

# GLOBAL STORE (important)
VECTOR_STORE = []   # [{chunk, embedding, file}]

# ================== AI ==================

def ask_llm(prompt):
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

# ================== CHUNKING ==================

def smart_chunk(text, size=400):
    lines = text.split("\n")
    chunks = []
    current = ""

    for line in lines:
        if len(current) + len(line) < size:
            current += line + "\n"
        else:
            chunks.append(current)
            current = line + "\n"

    if current:
        chunks.append(current)

    return chunks

# ================== EMBEDDINGS ==================

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def build_vector_store(results):
    global VECTOR_STORE
    VECTOR_STORE = []

    for r in results:
        chunks = smart_chunk(r["content"])

        embeddings = embedder.encode(chunks)

        for i in range(len(chunks)):
            VECTOR_STORE.append({
                "chunk": chunks[i],
                "embedding": embeddings[i],
                "file": r["file"]
            })

def retrieve(question, top_k=4):
    q_emb = embedder.encode([question])[0]

    scored = []

    for item in VECTOR_STORE:
        score = cosine_similarity(q_emb, item["embedding"])

        # slight boost for important files
        if "readme" in item["file"].lower():
            score += 0.1
        if item["file"].endswith(".ipynb"):
            score += 0.05

        scored.append((score, item))

    scored.sort(reverse=True)

    return [x[1] for x in scored[:top_k]]

# ================== EXPLAIN ==================

def explain_code(code, filename):
    return ask_llm(f"""
Explain this file:

File: {filename}

- Purpose
- Key logic

Keep it short.

Code:
{code[:1200]}
""")

# ================== SUMMARY ==================

def get_summary(results):
    context = ""
    for r in results:
        context += f"\nFILE: {r['file']}\n{r['content']}\n"

    return ask_llm(f"""
Summarize this project:

- One-line description
- 3 features
- Tech stack

Keep under 6 lines.

{context[:3000]}
""")

# ================== Q&A (OPTIMIZED) ==================

def ask_question(question):
    retrieved = retrieve(question)

    context = ""
    for item in retrieved:
        context += f"\nFILE: {item['file']}\n{item['chunk']}\n"

    return ask_llm(f"""
You are an expert code analyst.

Answer ONLY from context.

Context:
{context}

Question: {question}

Answer clearly:
""")

# ================== GITHUB ==================

def analyze_repo(repo_url):
    parts = repo_url.strip().rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"

    results = []
    state = {"count": 0, "limit": 6}

    def fetch(api):
        if state["count"] >= state["limit"]:
            return

        res = requests.get(api)
        data = res.json()

        for item in data:
            if state["count"] >= state["limit"]:
                return

            name = item["name"]

            if item["type"] == "file" and item["download_url"]:
                if not name.endswith(valid_extensions):
                    continue

                raw = requests.get(item["download_url"]).text

                if name.endswith(".ipynb"):
                    try:
                        notebook = json.loads(raw)
                        code = "\n".join(
                            "".join(cell["source"])
                            for cell in notebook["cells"]
                            if cell["cell_type"] == "code"
                        )
                    except:
                        code = raw
                else:
                    code = raw

                explanation = explain_code(code, name)

                results.append({
                    "file": name,
                    "content": code[:2000],
                    "explanation": explanation
                })

                state["count"] += 1

            elif item["type"] == "dir":
                fetch(item["url"])

    fetch(api_url)

    # 🔥 BUILD VECTOR STORE ONCE
    build_vector_store(results)

    return results