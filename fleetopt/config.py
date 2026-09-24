"""One place for how fleetopt configures itself and how it authenticates.

Credentials: fleetopt manages none. The optimizer and the judge both run through
the Claude Code binary bundled with the Agent SDK, so they authenticate exactly
the way Claude Code on this machine does - a claude.ai login, ANTHROPIC_API_KEY or
ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL in the environment or in the `env` block
of ~/.claude/settings.json, an apiKeyHelper, or the Bedrock/Vertex/Foundry
switches. If Claude Code works on the machine, fleetopt works. Nothing to copy.

Knobs (FLEETOPT_*): real environment, then ./.env, then ~/.config/fleetopt/env.
Same format everywhere, first hit wins.
"""

import json
import os
import pathlib
import shutil
import subprocess

GLOBAL_ENV = pathlib.Path.home() / ".config" / "fleetopt" / "env"

# Keys fleetopt itself put into the environment. The target's process gets the
# operator's real environment and none of these: a key in fleetopt's .env must
# never become the key the target bills to.
_injected = set()


def load_env():
    for path in (pathlib.Path(".env"), GLOBAL_ENV):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key not in os.environ:
                os.environ[key] = value.strip()
                _injected.add(key)


def child_env():
    """Environment for the target's process. See _injected."""
    return {k: v for k, v in os.environ.items() if k not in _injected}


# Which Claude Code settings the SDK sessions load. "user" keeps
# ~/.claude/settings.json, where a company puts its gateway URL and key and where
# apiKeyHelper lives. The target repo's own .claude/ (hooks, permissions,
# CLAUDE.md) is deliberately NOT loaded: it is another team's configuration and
# an instruction surface we don't control. Managed policy settings always apply.
SETTING_SOURCES = ["user"]

# Passed to every Claude Code subprocess fleetopt spawns.
SDK_ENV = {
    # No version checks, telemetry or release notes leaving the machine.
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    # No per-directory memory written or read; runs must not remember each other.
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
}

# The optimizer reads the target's source; it has no business reading its secrets.
# Read rules also cover Grep and Glob (best effort, per Claude Code's permissions).
DENY_READS = [
    f"Read({pattern})"
    for pattern in (
        "**/.env", "**/.env.*", "**/*.pem", "**/*.key", "**/id_rsa*", "**/id_ed25519*",
        "**/*secret*", "**/*credential*", "**/.netrc", "**/.npmrc", "**/.pypirc",
    )
]


def _bundled_cli():
    try:
        import claude_agent_sdk
    except ImportError:
        return None
    exe = pathlib.Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    return str(exe) if exe.exists() else None


def auth_summary():
    """What the Claude Code binary says it will authenticate with. Informational:
    the binary is the authority and fails clearly by itself if there is nothing."""
    exe = _bundled_cli() or shutil.which("claude")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "auth", "status"], capture_output=True, text=True, timeout=20)
        status = json.loads(out.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    provider = status.get("apiProvider")
    via = f" via {provider}" if provider and provider != "firstParty" else ""
    if status.get("apiKeySource"):
        return f"{status['apiKeySource']}{via}"
    if status.get("authMethod") == "oauth_token":
        return f"bearer token (ANTHROPIC_AUTH_TOKEN){via}"
    if status.get("loggedIn"):
        plan = status.get("subscriptionType") or "org"
        return f"claude.ai login ({plan}, {status.get('email', '?')}){via}"
    return ("no credential found - log in to Claude Code, or set ANTHROPIC_API_KEY / "
            "ANTHROPIC_AUTH_TOKEN (+ ANTHROPIC_BASE_URL) in the environment or in "
            "~/.claude/settings.json")
