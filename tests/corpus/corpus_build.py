"""Corpus of open-source LangGraph agents, from GitHub search rather than memory.

    python tests/corpus/corpus_build.py [--top 45] [--min-stars 30] [--days 365]

Writes corpus.md and corpus.json next to this file. Unauthenticated GitHub allows
10 search calls a minute and 60 other calls an hour, so tree inspection (which
is what tells us size and patterns) is capped by --top. Export GITHUB_TOKEN for
5000 calls an hour; the script then also reads each repo's dependency file to
name the LLM providers it pulls in.
"""

import argparse
import base64
import datetime as dt
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"
HERE = pathlib.Path(__file__).parent
TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
SKIP = {"langchain-ai/langgraph", "langchain-ai/langchain", "langchain-ai/langgraphjs"}

QUERIES = [
    "topic:langgraph language:python",
    "topic:langgraph-agent language:python",
    "topic:langgraph-agents language:python",
    "langgraph in:name,description language:python",
    "langgraph agent in:readme language:python",
]

# Applied to name + description + topics, and to every path in the repo tree.
PATTERNS = {
    "supervisor": r"supervisor|orchestrat|coordinator|hierarch",
    "swarm": r"swarm|handoff",
    "subgraphs": r"subgraph",
    "reflection": r"reflect|critic|self.?correct|revis|grader|evaluator",
    "router": r"router|routing|triage|classif|intent",
    "plan-execute": r"\bplan(ner|ning)?\b|planner",
    "RAG": r"\brag\b|retriev|vector|embedding|knowledge.?base",
    "memory": r"memor(y|ies)|checkpoint",
    "HITL": r"interrupt|human.?in.?the.?loop|hitl|approval",
    "MCP": r"\bmcp\b",
    "research": r"research",
}
FLAGS = {
    "collection": r"awesome|tutorial|course|examples?|cookbook|learn|notes|bootcamp|curriculum|template|starter|boilerplate",
    "library": r"librar|framework|\bsdk\b|toolkit|integration",
}
PROVIDERS = r"langchain[-_](anthropic|openai|google[-_]genai|google[-_]vertexai|aws|ollama|groq|mistralai|cohere|xai|deepseek|azure)"
NOISE_DIRS = re.compile(r"(^|/)(\.venv|venv|node_modules|site-packages|\.git|__pycache__|tests?|docs?)(/|$)")


def get(url, params=None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "fleetopt-corpus"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    for _ in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                reset = int(e.headers.get("X-RateLimit-Reset") or time.time() + 60)
                wait = min(max(reset - time.time() + 1, 5), 90)
                print(f"  rate limited, waiting {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if e.code == 404:
                return None
            raise
    raise RuntimeError(f"gave up on {url}")


def search(query, pages):
    found = {}
    for page in range(1, pages + 1):
        data = get(f"{API}/search/repositories", {"q": query, "sort": "stars", "order": "desc", "per_page": 100, "page": page})
        items = (data or {}).get("items", [])
        for it in items:
            found[it["full_name"]] = it
        if len(items) < 100:
            break
    return found


def signals(text):
    text = text.lower()
    return sorted(k for k, rx in PATTERNS.items() if re.search(rx, text))


def flags(text):
    text = text.lower()
    return sorted(k for k, rx in FLAGS.items() if re.search(rx, text))


def inspect(repo):
    """One tree call (plus one dependency read with a token). Returns size/pattern facts."""
    tree = get(f"{API}/repos/{repo['full_name']}/git/trees/{repo['default_branch']}", {"recursive": "1"})
    if not tree:
        return {"error": "no tree"}
    paths = [t["path"] for t in tree.get("tree", []) if t["type"] == "blob"]
    code = [p for p in paths if p.endswith(".py") and not NOISE_DIRS.search(p)]
    facts = {
        "py_files": len(code),
        "notebooks": sum(p.endswith(".ipynb") for p in paths),
        "langgraph_json": [p for p in paths if p.endswith("langgraph.json")],
        "tests": sum(bool(re.search(r"(^|/)tests?/|(^|/)test_[^/]*\.py$", p)) for p in paths),
        "evals": sorted({p for p in paths if re.search(r"eval|golden|dataset|deepeval|promptfoo|\.jsonl$", p, re.I)})[:6],
        "path_patterns": signals(" ".join(code)),
        "truncated": tree.get("truncated", False),
        "providers": [],
        "langgraph_dep": None,  # None = not checked (no token)
    }
    if TOKEN:
        facts["langgraph_dep"] = False
        for dep in ("pyproject.toml", "requirements.txt"):
            blob = get(f"{API}/repos/{repo['full_name']}/contents/{dep}")
            if blob and blob.get("content"):
                text = base64.b64decode(blob["content"]).decode("utf-8", "replace")
                facts["providers"] += sorted({m.group(1) for m in re.finditer(PROVIDERS, text)})
                facts["langgraph_dep"] |= bool(re.search(r"\blanggraph\b", text))
    return facts


def code_search_repos(pages):
    """Repos that ship a langgraph.json, i.e. run with `langgraph dev`, whatever
    their star count. Code search needs a token."""
    names = set()
    for page in range(1, pages + 1):
        data = get(f"{API}/search/code", {"q": "filename:langgraph.json", "per_page": 100, "page": page})
        items = (data or {}).get("items", [])
        names |= {it["repository"]["full_name"] for it in items}
        if len(items) < 100:
            break
    return names


def tier(py):
    return "S" if py <= 8 else "M" if py <= 30 else "L" if py <= 120 else "XL"


def clean(s, n=90):
    s = re.sub(r"[|\r\n]+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=300 if TOKEN else 45)
    ap.add_argument("--min-stars", type=int, default=30)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--pages", type=int, default=3)
    args = ap.parse_args()

    repos = {}
    for q in QUERIES:
        got = search(q, args.pages)
        print(f"{len(got):4d}  {q}", file=sys.stderr)
        repos.update(got)
    if TOKEN:
        extra = code_search_repos(args.pages * 3) - set(repos)
        print(f"{len(extra):4d}  filename:langgraph.json (repos not found by the searches above)", file=sys.stderr)
        for name in sorted(extra):
            r = get(f"{API}/repos/{name}")
            if r:
                repos[name] = r

    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.days)).isoformat()
    keep = [
        r for r in repos.values()
        if not r["archived"] and not r["fork"] and r["stargazers_count"] >= args.min_stars
        and r["pushed_at"] >= cutoff and r["full_name"] not in SKIP
    ]
    # Agent applications first, then link lists and libraries; stars break ties.
    # Star count alone surfaces infrastructure that merely mentions LangGraph.
    keep.sort(key=lambda r: (bool(flags(f"{r['full_name']} {r.get('description') or ''}")), -r["stargazers_count"]))
    print(f"{len(repos)} unique, {len(keep)} after filters, inspecting {min(args.top, len(keep))}", file=sys.stderr)

    rows = []
    for i, r in enumerate(keep):
        text = f"{r['full_name']} {r.get('description') or ''} {' '.join(r.get('topics') or [])}"
        row = {
            "repo": r["full_name"], "url": r["html_url"], "stars": r["stargazers_count"],
            "pushed": r["pushed_at"][:10], "description": clean(r.get("description")),
            "desc_patterns": signals(text), "flags": flags(text), "inspected": i < args.top,
        }
        if i < args.top:
            try:
                row.update(inspect(r))
            except Exception as exc:  # keep going; one bad repo must not lose the corpus
                row["error"] = str(exc)[:80]
            time.sleep(0.2)
        rows.append(row)

    (HERE / "corpus.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    def patterns(row):
        return ", ".join(sorted(set(row["desc_patterns"]) | set(row.get("path_patterns", [])))) or "-"

    inspected = [r for r in rows if r["inspected"] and "py_files" in r]
    order = {"XL": 0, "L": 1, "M": 2, "S": 3}
    inspected.sort(key=lambda r: (order[tier(r["py_files"])], -r["stars"]))
    rest = [r for r in rows if not r["inspected"] or "py_files" not in r]

    out = [
        f"# LangGraph agent corpus ({dt.date.today()})",
        "",
        f"GitHub search over {len(QUERIES)} queries (topics and keywords, Python), {len(repos)} unique repos.",
        f"Kept {len(keep)}: not archived, not a fork, >= {args.min_stars} stars, pushed in the last {args.days} days.",
        f"Inspected the top {len(inspected)} by stars with one tree call each: .py count outside tests/docs,",
        "pattern words in file paths and description, langgraph.json (runs with `langgraph dev`), tests, eval-like files.",
        "Size: S <= 8 py files, M <= 30, L <= 120, XL more. Patterns are keyword hits, a starting point for a human, not a verdict.",
        f"Providers column filled only with GITHUB_TOKEN set ({'yes' if TOKEN else 'no'}).",
        "",
        "## Inspected",
        "",
        "| Repo | Stars | Pushed | Size | .py | Patterns | langgraph.json | LG dep | Tests | Evals | Providers | Flags | What |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    dep = {None: "?", True: "yes", False: "no"}
    for r in inspected:
        out.append(
            f"| [{r['repo']}]({r['url']}) | {r['stars']} | {r['pushed']} | {tier(r['py_files'])} | {r['py_files']}"
            f"{' (+' + str(r['notebooks']) + ' nb)' if r['notebooks'] else ''} | {patterns(r)}"
            f" | {'yes' if r['langgraph_json'] else '-'} | {dep[r.get('langgraph_dep')]} | {r['tests'] or '-'} | {len(r['evals']) or '-'}"
            f" | {', '.join(r['providers']) or '-'} | {', '.join(r['flags']) or '-'} | {r['description']} |"
        )
    out += ["", f"## Not inspected ({len(rest)}, next in line by stars)", "", "| Repo | Stars | Pushed | Patterns (description) | Flags | What |", "|---|---|---|---|---|---|"]
    for r in rest:
        out.append(f"| [{r['repo']}]({r['url']}) | {r['stars']} | {r['pushed']} | {', '.join(r['desc_patterns']) or '-'} | {', '.join(r['flags']) or '-'} | {r['description']} |")
    (HERE / "corpus.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {HERE / 'corpus.md'} ({len(inspected)} inspected, {len(rest)} listed)", file=sys.stderr)


if __name__ == "__main__":
    main()
