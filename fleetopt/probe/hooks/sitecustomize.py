"""Injected into the target process via PYTHONPATH.

Python's `site` module imports this automatically at interpreter startup, which
is how we instrument a project without editing a single line of its source.

Keep this file import-light and failure-tolerant: it runs inside someone else's
process, and nothing here is allowed to break their program.
"""

import os
import sys


def _chain_original():
    """Run any sitecustomize the target project shipped, which we shadowed."""
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == here:
            continue
        candidate = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(candidate):
            try:
                with open(candidate) as fh:
                    source = fh.read()
                exec(  # noqa: S102 - deliberately running the file we displaced
                    compile(source, candidate, "exec"),
                    {"__name__": "sitecustomize", "__file__": candidate},
                )
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[fleetopt] chained sitecustomize failed: {exc}", file=sys.stderr)
            return


if os.environ.get("FLEETOPT_CAPTURE"):
    try:
        import _fleetopt_hook

        _fleetopt_hook.install()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[fleetopt] capture disabled: {exc}", file=sys.stderr)

_chain_original()
