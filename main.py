import requests
import os
from openai import OpenAI

# ================== CONFIG ==================

# use environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

valid_extensions = (".py", ".js", ".html", ".htm", ".css", ".md", ".txt")

# ================== AI FUNCTION ==================

def explain_code(code):
    prompt = f"""
    Explain this code in simple terms for a beginner:

    {code[:1000]}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# ================== FETCH FUNCTION ==================

def fetch_files(api_url, count=0):
    response = requests.get(api_url)

    if response.status_code != 200:
        print("Error:", response.status_code)
        return

    data = response.json()

    if isinstance(data, dict) and "message" in data:
        print("GitHub Error:", data["message"])
        return

    for item in data:

        if item["type"] == "file" and item["download_url"]:

            if not item["name"].endswith(valid_extensions):
                continue

            print(f"\n--- {item['name']} ---")

            try:
                content = requests.get(item["download_url"]).text

                explanation = explain_code(content)

                print("\nExplanation:\n", explanation)

                count += 1

                if count == 2:   # limit to avoid API cost
                    return

            except Exception as e:
                print("Error reading file:", e)

        elif item["type"] == "dir":
            fetch_files(item["url"], count)


# ================== MAIN ==================

repo_url = input("Enter GitHub repo URL: ").strip().rstrip("/")

parts = repo_url.split("/")

if len(parts) < 5:
    print("Invalid URL. Use: https://github.com/owner/repo")
    exit()

owner = parts[-2]
repo = parts[-1]

api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"

print("\n📂 AI Codebase Explainer Running...\n")

fetch_files(api_url)