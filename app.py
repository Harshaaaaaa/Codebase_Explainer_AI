from flask import Flask, render_template, request
from analyzer import analyze_repo, ask_question, get_summary
import os

app = Flask(__name__)

results_cache = []
summary_cache = ""

@app.route("/", methods=["GET", "POST"])
def home():
    global results_cache, summary_cache

    if request.method == "POST":
        repo = request.form["repo"]

        results_cache = analyze_repo(repo)
        summary_cache = get_summary(results_cache)

        return render_template("index.html", results=results_cache, summary=summary_cache)

    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]
    answer = ask_question(results_cache, question)
    return {"answer": answer}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))