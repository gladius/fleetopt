"""Structural facts about a graph, from what it declared and what it actually did.

The architecture review is only worth having if every finding is a number. This
module computes those numbers from the capture database and nothing else: the
declared graph (nodes, edges, which edges are conditional) and the node sequence
of every captured trace. No model call, no opinion. The reviewer skill turns
these into findings; a human turns findings into decisions.

Smells computed:
- branches declared but never taken, and branch points that always go one way;
- a branch point whose observed order of targets is identical in every trace
  (a dispatcher that is really a fixed pipeline), flagged harder when the node
  also calls a model to make that non-decision;
- a node that runs the same number of times (>1) in every trace: a loop that runs
  to its cap instead of exiting early;
- a repeated model-calling node whose replies are identical round after round.
"""

import json
from collections import Counter, defaultdict


def _in(ids):
    return f"({','.join('?' * len(ids))})"


def declared(conn, session_ids):
    """Nodes and edges as compiled, from the newest graph snapshot in these sessions."""
    row = conn.execute(
        f"SELECT nodes, edges FROM graphs WHERE session_id IN {_in(session_ids)}"
        " ORDER BY session_id DESC LIMIT 1", session_ids,
    ).fetchone()
    if not row:
        return [], []
    return json.loads(row["nodes"] or "[]"), json.loads(row["edges"] or "[]")


def node_sequences(conn, session_ids):
    """{trace_id: [(step, node), ...]} from node runs (a chain run named after its node)."""
    seqs = defaultdict(list)
    for r in conn.execute(
        f"SELECT trace_id, step, node FROM runs WHERE session_id IN {_in(session_ids)}"
        " AND run_type = 'chain' AND node IS NOT NULL AND name = node AND step IS NOT NULL"
        " ORDER BY trace_id, step", session_ids,
    ):
        seqs[r["trace_id"]].append((r["step"], r["node"]))
    return dict(seqs)


def llm_replies(conn, session_ids):
    """{trace_id: {node: [completion, ...]}} for every model call."""
    out = defaultdict(lambda: defaultdict(list))
    for r in conn.execute(
        f"SELECT trace_id, node, completion FROM runs WHERE session_id IN {_in(session_ids)}"
        " AND run_type = 'llm' ORDER BY trace_id, step", session_ids,
    ):
        out[r["trace_id"]][r["node"]].append(r["completion"] or "")
    return out


def _transitions(seq):
    """Observed (src, dst) pairs, step to next step; parallel nodes share a step."""
    by_step = defaultdict(list)
    for step, node in seq:
        by_step[step].append(node)
    steps = sorted(by_step)
    pairs = []
    for a, b in zip(steps, steps[1:]):
        pairs += [(s, d) for s in by_step[a] for d in by_step[b]]
    return pairs


def analyze(conn, session_ids):
    nodes, edges = declared(conn, session_ids)
    seqs = node_sequences(conn, session_ids)
    replies = llm_replies(conn, session_ids)
    n = len(seqs)
    findings = []
    if not n:
        return {"traces": 0, "nodes": nodes, "findings": findings}

    llm_nodes = {node for per in replies.values() for node in per}
    real_edges = [e for e in edges if e["target"] != "__end__" and e["source"] != "__start__"]
    cond_targets = defaultdict(set)
    for e in real_edges:
        if e.get("conditional"):
            cond_targets[e["source"]].add(e["target"])

    # Which declared transitions were taken, in how many traces.
    taken = Counter()
    order_per_source = defaultdict(list)  # source -> [tuple of targets in order, per trace]
    for seq in seqs.values():
        pairs = _transitions(seq)
        for pair in set(pairs):
            taken[pair] += 1
        for src in cond_targets:
            # Self-loops are the loop's own rounds (see constant_rounds), not a dispatch order.
            order_per_source[src].append(tuple(d for s, d in pairs if s == src and d != src))

    for src, targets in sorted(cond_targets.items()):
        counts = {t: taken[(src, t)] for t in sorted(targets)}
        never = [t for t, c in counts.items() if c == 0]
        always = [t for t, c in counts.items() if c == n]
        if never:
            findings.append({
                "kind": "branch_never_taken", "node": src, "targets": never, "traces": n,
                "text": f"{src}: {len(targets)} branches declared, never taken in {n} traces: {', '.join(never)}"
                        + (f"; always goes to {', '.join(always)}" if always else ""),
            })
        orders = order_per_source[src]
        if orders and len(set(orders)) == 1 and len(orders[0]) > 1:
            fixed = orders[0]
            with_model = src in llm_nodes
            findings.append({
                "kind": "fixed_dispatch", "node": src, "order": list(fixed), "traces": n, "calls_model": with_model,
                "text": f"{src}: dispatches {' -> '.join(fixed)} in the same order in {n}/{n} traces"
                        + (" while calling a model at every step to decide" if with_model else ""),
            })

    # Loops that always run the same number of rounds.
    rounds = defaultdict(list)
    for seq in seqs.values():
        for node, c in Counter(node for _, node in seq).items():
            rounds[node].append(c)
    for node, counts in sorted(rounds.items()):
        if len(counts) == n and len(set(counts)) == 1 and counts[0] > 1:
            findings.append({
                "kind": "constant_rounds", "node": node, "rounds": counts[0], "traces": n,
                "text": f"{node}: ran exactly {counts[0]} times in every one of {n} traces (runs to its cap, never exits early)",
            })

    # Repeated model calls whose replies never change within a trace.
    for node in sorted(llm_nodes):
        per_trace = [per[node] for per in replies.values() if len(per.get(node, [])) > 1]
        if per_trace and len(per_trace) == n and all(len(set(r)) == 1 for r in per_trace):
            findings.append({
                "kind": "repeated_identical_reply", "node": node, "traces": n,
                "text": f"{node}: the model's reply is identical across all rounds in {n}/{n} traces",
            })

    return {
        "traces": n,
        "nodes": [x for x in nodes if not x.startswith("__")],
        "llm_nodes": sorted(llm_nodes),
        "conditional_sources": sorted(cond_targets),
        "findings": findings,
    }


def render(result):
    if not result["traces"]:
        return "no traces to analyze"
    head = (f"{result['traces']} traces, {len(result['nodes'])} nodes, {len(result['llm_nodes'])} call a model,"
            f" {len(result['conditional_sources'])} branch points")
    if not result["findings"]:
        return head + "\nno structural smells on these traces"
    return head + "\n" + "\n".join(f"- {f['text']}" for f in result["findings"])
