import requests
import json
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

valid_extensions = (".py", ".ipynb", ".md")

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


def explain_code(code):
    return ask_llm(f"""
Explain this code briefly:

- Purpose
- Key functions
- Logic

Code:
{code[:1200]}
""")


def get_summary(results):
    context = ""
    for r in results:
        context += r["content"] + "\n"

    return ask_llm(f"""
Summarize this project in SHORT format:

- One-line description
- 3 key features
- Tech stack (if visible)

Keep it under 6 lines.

Code:
{context[:3000]}
""")


def ask_question(results, question):
    # simple relevance filter
    relevant = [
        r for r in results
        if any(word in r["content"].lower() for word in question.lower().split())
    ]

    context = ""
    for r in relevant[:2]:
        context += r["content"]

    return ask_llm(f"""
Answer based on this code:

{context}

Question: {question}
""")


# ================== GITHUB ==================

def analyze_repo(repo_url):
    parts = repo_url.strip().rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"

    results = []
    state = {"count": 0, "limit": 5}

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

                explanation = explain_code(code)

                results.append({
                    "file": name,
                    "content": code[:1200],
                    "explanation": explanation
                })

                state["count"] += 1

            elif item["type"] == "dir":
                fetch(item["url"])

    fetch(api_url)

    return results