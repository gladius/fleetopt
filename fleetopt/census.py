"""The census: what a project holds, taken by code, exhaustively, before any model sees it.

A session that finds things at its own discretion stops looking when it thinks it has seen
enough (observed: an agent in a sub-folder, its env file beside it, both missed). The census
walks the whole tree and lists every agent, env file, launcher, eval and external system
with its path, so what is handed to the surveyor is a map it cannot skip. Names and
numbers only: never an env file's values, never a file's contents.
"""

import ast
import collections
import hashlib
import json
import os
import pathlib
import re

SKIP = {".git", ".fleetopt", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}
LAUNCHERS = ("Dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml", "Makefile",
             ".envrc", "Procfile", "langgraph.json", "pyproject.toml", "requirements.txt", "uv.lock", "poetry.lock")
EVALS = re.compile(r"(^|/)(tests?|evals?|evaluations?|datasets?|eval_data|goldens?|fixtures?|examples?)(/|$)"
                   r"|(^|/)(test_[^/]*|[^/]*_test|conftest)\.py$"
                   r"|(^|/)[^/]*(inputs?|requests?|samples?|examples?|goldens?|evals?|cases?)[^/]*\.(jsonl?|csv|ya?ml|txt|md)$", re.I)
# What a package reaches outside the process: the external systems an agent depends on.
SYSTEMS = {
    "httpx": "HTTP calls", "requests": "HTTP calls", "aiohttp": "HTTP calls", "boto3": "AWS", "botocore": "AWS",
    "langchain_aws": "AWS Bedrock", "google": "Google Cloud / Gemini", "langchain_google_genai": "Gemini",
    "langchain_google_vertexai": "Vertex AI", "azure": "Azure", "langchain_openai": "OpenAI", "openai": "OpenAI",
    "anthropic": "Anthropic", "langchain_anthropic": "Anthropic", "psycopg": "SQL database", "psycopg2": "SQL database",
    "asyncpg": "SQL database", "sqlalchemy": "SQL database", "pymysql": "SQL database", "redis": "Redis",
    "pymongo": "MongoDB", "motor": "MongoDB", "pinecone": "vector store", "qdrant_client": "vector store",
    "chromadb": "vector store", "weaviate": "vector store", "faiss": "vector store", "slack_sdk": "Slack",
    "smtplib": "email", "tavily": "web search", "serpapi": "web search", "duckduckgo_search": "web search",
    "langsmith": "LangSmith", "mcp": "MCP servers", "langchain_mcp_adapters": "MCP servers", "kafka": "message queue",
    "pika": "message queue", "elasticsearch": "search index", "opensearchpy": "search index", "snowflake": "Snowflake",
    "databricks": "Databricks", "jira": "Jira", "github": "GitHub", "twilio": "Twilio",
}


def walk(project):
    """Every file under the project, as a path relative to it, the skipped folders left out."""
    project = pathlib.Path(project)
    for folder, dirs, files in os.walk(project):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith("."))
        for name in sorted(files):
            yield (pathlib.Path(folder) / name).relative_to(project).as_posix()


def _python(project, rel):
    """(packages it imports, whether it builds a LangGraph graph, the name of what `adk run`
    would start in it: `app` first, else `root_agent`, else None)."""
    try:
        tree = ast.parse((project / rel).read_text(encoding="utf-8-sig", errors="replace"))
    except (SyntaxError, ValueError, OSError):
        return set(), False, None
    packages, builds, names = set(), False, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            packages |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            packages.add("google.adk" if node.module.startswith("google.adk") else node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "compile":
            builds = True
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name) and t.id in ("app", "root_agent")}
    adk = "google.adk" in packages and ("app" if "app" in names else "root_agent" if "root_agent" in names else None)
    return packages, builds and "langgraph" in packages, adk or None


def env_names(project):
    """({env file: the names it sets}, credential names set in this terminal), for every env
    file in the project, each name marked when it is empty or set by a command. Names only: a
    value is never kept or shown."""
    project = pathlib.Path(project)

    def state(name, value):
        value = value.strip().strip("'\"")
        return name + (" (empty)" if not value else " (set by a command)" if value.startswith(("$(", "`")) else "")

    files = {}
    for rel in walk(project):
        name = rel.rsplit("/", 1)[-1]
        if not (name == ".env" or name.startswith(".env.") or name.endswith(".env")):
            continue
        lines = (project / rel).read_text(encoding="utf-8-sig", errors="replace").splitlines()
        files[rel] = [state(key, value) for line in lines if "=" in line and not line.strip().startswith("#")
                      for key, _, value in [line.strip().removeprefix("export ").partition("=")] if key.strip()]
    secret = re.compile(r"key|token|secret|password|credential", re.I)
    return files, sorted(state(k, v) for k, v in os.environ.items() if secret.search(k) and not k.startswith("FLEETOPT_"))


def interpreters(project):
    """The python of every virtual environment in the project, relative to it, nearest the
    root first."""
    project = pathlib.Path(project)
    found = []
    for folder, dirs, _ in os.walk(project):
        for d in sorted(dirs):
            if d in ("venv", ".venv"):
                for rel in ("bin/python", "Scripts/python.exe"):
                    if (pathlib.Path(folder) / d / rel).exists():
                        found.append((pathlib.Path(folder) / d / rel).relative_to(project).as_posix())
        dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")]
    return sorted(found, key=lambda p: p.count("/"))


def take(project):
    """The whole map, as plain data."""
    project = pathlib.Path(project).resolve()
    files = list(walk(project))
    kinds = collections.Counter(pathlib.Path(f).suffix or "(none)" for f in files)
    graphs, systems, builders, adk, launchers, readmes, evals = {}, collections.defaultdict(set), [], [], [], [], []
    for rel in files:
        name = rel.rsplit("/", 1)[-1]
        if name == "langgraph.json":
            try:
                graphs[rel] = json.loads((project / rel).read_text(encoding="utf-8-sig")).get("graphs") or {}
            except (ValueError, AttributeError, OSError):
                graphs[rel] = {}
        if name in LAUNCHERS:
            launchers.append(rel)
        if name.lower().startswith("readme"):
            readmes.append(rel)
        if EVALS.search(rel):
            evals.append(rel)
        if rel.endswith(".py"):
            packages, builds, root = _python(project, rel)
            if builds:
                builders.append(rel)
            if root:
                adk.append(f"{rel}:{root}")
            for p in packages & SYSTEMS.keys():
                systems[SYSTEMS[p]].add(rel)
    env, shell = env_names(project)
    return {"project": str(project), "files": len(files), "kinds": dict(kinds.most_common(8)),
            "graphs": graphs, "builders": builders, "adk_agents": adk, "launchers": launchers, "readmes": readmes,
            "env_files": env, "shell_credentials": shell, "interpreters": interpreters(project),
            "systems": {k: sorted(v) for k, v in sorted(systems.items())}, "evals": evals}


def render(census, cap=40):
    """The map in words, for a session: lists capped, so a huge project stays a map."""
    some = lambda items: ", ".join(items[:cap]) + (f" ... and {len(items) - cap} more" if len(items) > cap else "") or "none"
    lines = [f"Files: {census['files']} ({', '.join(f'{k} {n}' for k, n in census['kinds'].items())})",
             "Agents declared (langgraph.json): " + ("; ".join(f"{p}: " + ", ".join(f"{n} = {s}" for n, s in g.items())
                                                              for p, g in census["graphs"].items()) or "none"),
             f"Files that build a LangGraph graph: {some(census['builders'])}",
             f"ADK agents (what `adk run` starts: an app, else a root_agent): {some(census.get('adk_agents') or [])}",
             f"Launchers and setup: {some(census['launchers'])}",
             f"READMEs: {some(census['readmes'])}",
             "Env files and the names each sets (never the values): "
             + ("; ".join(f"{f}: {', '.join(n[:cap]) or 'nothing'}" for f, n in list(census["env_files"].items())[:cap]) or "none"),
             f"Credential names set in this terminal: {some(census['shell_credentials'])}",
             f"Virtual environments: {some(census['interpreters'])}",
             "External systems, by the packages the code imports: "
             + ("; ".join(f"{s}: {', '.join(f[:6])}" for s, f in census["systems"].items()) or "none seen"),
             f"Tests, evals, data and examples: {some(census['evals'])}"]
    return "\n".join(lines)


def frameworks(census):
    """Which frameworks the project builds on, by what the census found: a skill named after
    one goes to an expert only on a project that uses it."""
    return ({"langgraph"} if census["graphs"] or census["builders"] else set()) | ({"adk"} if census["adk_agents"] else set())


def fingerprint(census):
    """What the survey depends on: the project's files, nothing of the machine (its path, what
    the terminal has set), so the survey holds on the next machine and goes stale when the
    project changes."""
    own = {k: v for k, v in census.items() if k not in ("project", "shell_credentials")}
    return hashlib.sha1(json.dumps(own, sort_keys=True).encode("utf-8")).hexdigest()[:12]
