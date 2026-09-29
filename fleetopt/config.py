"""One place for how fleetopt configures itself and how it authenticates.

Credentials: fleetopt manages none. The optimizer and the judge both run through
the Claude Code binary bundled with the Agent SDK, so they authenticate exactly
the way Claude Code on this machine does - a claude.ai login, ANTHROPIC_API_KEY or
ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL in the environment or in the `env` block
of ~/.claude/settings.json, an apiKeyHelper, or the Bedrock/Vertex/Foundry
switches. If Claude Code works on the machine, fleetopt works. Nothing to copy.
Only those credential keys are taken from settings.json; the operator's plugins,
skills, hooks and MCP servers stay out of every run (SETTING_SOURCES, sdk_args).

Knobs (FLEETOPT_*): real environment, then ./.env, then ~/.config/fleetopt/env.
Same format everywhere, first hit wins.
"""

import json
import os
import pathlib

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


# Which Claude Code settings the SDK sessions load: none. Loading "user" would pull
# in the operator's own plugins, skills, hooks and MCP servers (measured on one dev
# box: 3 plugins, 35 skills, 5 servers including Gmail and Drive, and 40x the
# fixed per-turn cost). A run on another team's repo gets none of that. What a
# company puts in ~/.claude/settings.json to reach its gateway - the `env` block
# and apiKeyHelper - is passed through explicitly by sdk_args(). The target repo's
# own .claude/ is not loaded either: another team's instruction surface. Managed
# policy settings and the claude.ai login apply regardless.
SETTING_SOURCES = []


# The settings.json keys that carry or fetch a credential. Everything else in that
# file (plugins, hooks, permissions, MCP servers, model choice) stays out of a run. A
# setup that depends on a key missing from this list fails loudly at the first model
# call rather than silently.
AUTH_KEYS = ("env", "apiKeyHelper", "awsAuthRefresh", "awsCredentialExport")


def _user_settings_path():
    root = os.environ.get("CLAUDE_CONFIG_DIR")
    return (pathlib.Path(root) if root else pathlib.Path.home() / ".claude") / "settings.json"


def sdk_args():
    """Extra CLI args for every SDK session: the credential-bearing keys of the
    user's settings.json (`env`, `apiKeyHelper`) as an inline --settings JSON, and
    nothing else from that file."""
    try:
        settings = json.loads(_user_settings_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    keep = {k: settings[k] for k in AUTH_KEYS if k in settings}
    return {"settings": json.dumps(keep)} if keep else {}

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
