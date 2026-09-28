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
- a repeated model-calling node whose replies are identical round after round;
- a node where something raised (and whether the node swallowed it), a node that
  paused for a human.
"""

import json
from collections import Counter, defaultdict


def _in(ids):
    return f"({','.join('?' * len(ids))})"


def declared(conn, session_ids, ran=()):
    """Nodes and edges as compiled. An agent built from other agents compiles several
    graphs; the numbers are about one of them, the one that accounts for the most of
    what actually ran (ties go to the larger graph)."""
    best, rank = ([], []), (-1, -1)
    for row in conn.execute(
        f"SELECT nodes, edges FROM graphs WHERE session_id IN {_in(session_ids)} ORDER BY session_id DESC", session_ids,
    ):
        nodes = json.loads(row["nodes"] or "[]")
        score = (len(set(nodes) & set(ran)), len(nodes))
        if score > rank:
            best, rank = (nodes, json.loads(row["edges"] or "[]")), score
    return best


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


def distinct_inputs(conn, session_ids, trace_ids):
    """How many different inputs the traces cover. A baseline is the same command run
    n times, so 6 traces are often 2 inputs: consistency, not variety. Inputs that
    carry generated ids or timestamps count as different; that overstates variety,
    never understates it, so treat the number as an upper bound."""
    seen = {
        r["trace_id"]: " ".join((r["inputs"] or "").split())
        for r in conn.execute(
            f"SELECT trace_id, inputs FROM runs WHERE session_id IN {_in(session_ids)}"
            " AND parent_run_id IS NULL", session_ids)
    }
    return len({seen.get(t) or t for t in trace_ids})


def calls_per_tool_round(conn, session_ids, nodes=()):
    """Model calls spent per round of tool use. A tool-calling agent spends one per
    round, plus one to give the answer: the same call picks the tool, writes its
    arguments and decides whether to stop. A graph that asks one model call what to do,
    another to do it and a third whether it is done spends three per round.

    Counted per trace, over traces that used a tool, leaving out one call per trace for
    the final answer. (The first version divided all calls by all rounds and called a
    plain two-node agent over-built: traces that needed no tool inflated it.)

    Only this graph's own nodes are counted. An agent nested inside one of them has
    its own loop, and its calls are not this graph's waste."""
    own = set(nodes)
    llm, rounds, nodes = Counter(), defaultdict(set), set()
    for r in conn.execute(
        f"SELECT trace_id, run_type, node, step FROM runs WHERE session_id IN {_in(session_ids)}"
        " AND run_type IN ('llm', 'tool')", session_ids,
    ):
        if own and r["node"] not in own:
            continue
        if r["run_type"] == "llm":
            llm[r["trace_id"]] += 1
            nodes.add(r["node"])
        else:
            rounds[r["trace_id"]].add(r["step"])
    calls = sum(max(llm[t] - 1, 0) for t in rounds)
    total = sum(len(steps) for steps in rounds.values())
    if not total or calls / total < 2:
        return []
    ratio = calls / total
    return [{
        "kind": "calls_per_tool_round", "node": "(graph)", "model_calls": calls, "tool_rounds": total,
        "traces": len(rounds), "ratio": round(ratio, 1), "model_nodes": sorted(x for x in nodes if x),
        "text": f"(graph): {calls} model calls for {total} rounds of tool use in {len(rounds)} traces, not counting"
                f" each trace's final answer ({ratio:.1f} per round), spread over {', '.join(sorted(x for x in nodes if x))};"
                " a tool-calling agent spends one per round",
    }]


def node_failures(conn, session_ids, n):
    """Nodes where something raised, and nodes that paused for a human. An interrupt
    is LangGraph's human-in-the-loop mechanism, not a failure. An error on a call
    inside a node whose own run did not fail was caught there: the graph went on
    without that result, which is worth more attention than a crash."""
    raised = defaultdict(lambda: {"traces": set(), "message": "", "node_failed": False})
    for r in conn.execute(
        f"SELECT trace_id, node, name, run_type, error FROM runs WHERE session_id IN {_in(session_ids)}"
        " AND error IS NOT NULL ORDER BY start_time", session_ids,
    ):
        text = r["error"] or ""
        cls = text.split("(", 1)[0].strip() or "Error"
        node = r["node"] or "(graph)"
        kind = "interrupt" if cls == "GraphInterrupt" else "node_error"
        entry = raised[(kind, node, cls)]
        entry["traces"].add(r["trace_id"])
        entry["message"] = entry["message"] or " ".join(text.split("Traceback", 1)[0].split())[:220]
        entry["node_failed"] |= r["run_type"] == "chain" and r["name"] in (r["node"], "LangGraph")
    out = []
    for (kind, node, cls), e in sorted(raised.items(), key=lambda kv: (kv[0][0] != "node_error", kv[0][1])):
        k = len(e["traces"])
        if kind == "interrupt":
            text = f"{node}: paused for a human in {k}/{n} traces"
        else:
            caught = "" if e["node_failed"] else " - caught inside the node, the graph continued without it"
            text = f"{node}: raised {cls} in {k}/{n} traces{caught}: {e['message']}"
        out.append({"kind": kind, "node": node, "error": cls, "traces": n, "count": k,
                    "swallowed": kind == "node_error" and not e["node_failed"], "text": text})
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
    seqs = node_sequences(conn, session_ids)
    nodes, edges = declared(conn, session_ids, {node for seq in seqs.values() for _, node in seq})
    if nodes:  # steps of agents nested inside this one are theirs, not this graph's
        seqs = {t: [(step, node) for step, node in seq if node in nodes] for t, seq in seqs.items()}
        seqs = {t: seq for t, seq in seqs.items() if seq}
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

    ran = {node for seq in seqs.values() for _, node in seq}
    for src, targets in sorted(cond_targets.items()):
        if src not in ran:
            continue  # it never ran, so it never had a branch to take; the node that skipped it is the finding
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
        # A dispatcher chooses among different targets. The same target over and over is
        # an agent loop going round (model -> tools -> model), which is rounds, not dispatch.
        if n > 1 and orders and len(set(orders)) == 1 and len(set(orders[0])) > 1:
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
        if n > 1 and len(counts) == n and len(set(counts)) == 1 and counts[0] > 1:
            findings.append({
                "kind": "constant_rounds", "node": node, "rounds": counts[0], "traces": n,
                "text": f"{node}: ran exactly {counts[0]} times in every one of {n} traces"
                        " (a cap it always reaches, or a fixed schedule; check which)",
            })

    # Repeated model calls whose replies never change within a trace.
    for node in sorted(llm_nodes):
        per_trace = [per[node] for per in replies.values() if len(per.get(node, [])) > 1]
        if n > 1 and per_trace and len(per_trace) == n and all(len(set(r)) == 1 for r in per_trace):
            findings.append({
                "kind": "repeated_identical_reply", "node": node, "traces": n,
                "text": f"{node}: the model's reply is identical across all rounds in {n}/{n} traces",
            })

    findings += calls_per_tool_round(conn, session_ids, nodes)
    findings = node_failures(conn, session_ids, n) + findings  # a node that raises comes first

    return {
        "traces": n,
        "distinct_inputs": distinct_inputs(conn, session_ids, list(seqs)),
        "nodes": [x for x in nodes if not x.startswith("__")],
        "llm_nodes": sorted(llm_nodes),
        "conditional_sources": sorted(cond_targets),
        "findings": findings,
    }


def render(result):
    if not result["traces"]:
        return "no traces to analyze"
    k = result.get("distinct_inputs", result["traces"])
    head = (f"{result['traces']} traces over at most {k} distinct inputs, {len(result['nodes'])} nodes,"
            f" {len(result['llm_nodes'])} call a model, {len(result['conditional_sources'])} branch points")
    if k < result["traces"]:
        head += (f"\nnote: the traces repeat the same inputs, so n/n counts show consistency, not variety;"
                 f" the evidence is {k} inputs wide")
    if not result["findings"]:
        return head + "\nno structural smells on these traces"
    return head + "\n" + "\n".join(f"- {f['text']}" for f in result["findings"])
