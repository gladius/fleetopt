"""Structural facts about a graph, from what it declared and what it actually did.

A design review is only worth having if every finding is a number. This module computes
those numbers from the recordings and nothing else: the declared graph (nodes, edges, which
edges are conditional) and the order its nodes ran in, every time it was invoked. No model
call, no opinion. The design expert turns them into findings; a person turns findings into
decisions.

Computed, for the graph the driver ran and for each graph nested in it, each on its own:
- branches declared but never taken, and branch points that always go one way;
- a branch point that sends the work to the same targets in the same order every time (a
  dispatcher that is really a fixed pipeline), flagged harder when the node also calls a
  model to make that non-decision;
- a node that runs the same number of times (>1) every time: a loop that runs to its cap
  instead of exiting early;
- a repeated model-calling node whose replies are identical round after round;
- two or more model calls spent per round of tool use;
- a node where something raised (and whether the node swallowed it), a node that paused for
  a human;
- a branch point whose targets the graph does not declare: what it never took cannot be
  counted, and the function has to be read.
"""

import json
from collections import Counter, defaultdict

ENTRY = "__start__"


def _in(ids):
    return f"({','.join('?' * len(ids))})"


def declared(conn, session_ids, ran=()):
    """(nodes, edges) as compiled, of the graph that ran: the one the driver marked. Without a
    mark (a recording made before there was one), the one that accounts for the most of what
    ran, the larger on a tie."""
    best, rank = ([], []), (False, -1, -1)
    for row in conn.execute(
            f"SELECT driven, nodes, edges FROM graphs WHERE session_id IN {_in(session_ids)} ORDER BY session_id DESC",
            session_ids):
        nodes = json.loads(row["nodes"] or "[]")
        score = (bool(row["driven"]), len(set(nodes) & set(ran)), len(nodes))
        if score > rank:
            best, rank = (nodes, json.loads(row["edges"] or "[]")), score
    return best


def _at(prefix, name):
    """A declared name as one graph sees it: its own node, or the nested graph the name is
    inside, which to this graph is one node. None when the name is not in this graph."""
    return name[len(prefix):].split(":")[0] if name.startswith(prefix) else None


def levels(nodes, edges):
    """{prefix: (its nodes, its edges)} for the graph ('' prefix) and each graph nested in it
    ('team:'). An edge inside a nested graph is that graph's, not its parent's."""
    prefixes = {""} | {n.rsplit(":", 1)[0] + ":" for n in nodes if ":" in n}
    out = {}
    for prefix in sorted(prefixes):
        own = sorted({_at(prefix, n) for n in nodes if n.startswith(prefix)})
        seen = []
        for e in edges:
            s, t = _at(prefix, e["source"]), _at(prefix, e["target"])
            deeper = ":" in e["source"][len(prefix):] and ":" in e["target"][len(prefix):]
            if s is None or t is None or (s == t and deeper):
                continue
            seen.append({"source": s, "target": t, "conditional": bool(e.get("conditional"))})
        out[prefix] = (own, seen)
    return out


def distinct_inputs(conn, session_ids):
    """How many different inputs the traces cover. A measurement is the same requests run n
    times, so 6 traces are often 2 inputs: consistency, not variety. Inputs that carry
    generated ids or timestamps count as different; that overstates variety, never
    understates it, so treat the number as an upper bound."""
    return len({" ".join((r["inputs"] or "").split()) or r["trace_id"] for r in conn.execute(
        f"SELECT trace_id, inputs FROM runs WHERE session_id IN {_in(session_ids)} AND parent_run_id IS NULL",
        session_ids)})


def _transitions(seq):
    """Observed (src, dst) pairs, step to next step; parallel nodes share a step."""
    by_step = defaultdict(list)
    for step, node in seq:
        by_step[step].append(node)
    steps = sorted(by_step)
    return [(s, d) for a, b in zip(steps, steps[1:]) for s in by_step[a] for d in by_step[b]]


def _level(prefix, nodes, edges, runs):
    """The findings of one graph, from every time it was invoked."""
    show = lambda node: "the entry" if node == ENTRY else prefix + node
    here = lambda r: r["path"] == prefix + (r["node"] or "") and r["node"] in nodes
    seqs = defaultdict(list)  # one sequence each time the graph was invoked: its node runs share a parent
    for r in runs:
        if r["run_type"] == "chain" and r["name"] == r["node"] and r["step"] is not None and here(r):
            seqs[r["parent_run_id"]].append((r["step"], r["node"]))
    n = len(seqs)
    if not n:
        return 0, []
    replies = defaultdict(lambda: defaultdict(list))  # {trace: {node: [reply, ...]}}
    for r in runs:
        if r["run_type"] == "llm" and here(r):
            replies[r["trace_id"]][r["node"]].append(r["completion"] or "")
    llm_nodes = {node for per in replies.values() for node in per}

    cond = defaultdict(set)  # a branch point, and the targets it declares
    for e in edges:
        if e["conditional"]:
            cond[e["source"]].update({e["target"]} - {"__end__"})
    taken, orders = Counter(), defaultdict(list)
    for seq in seqs.values():
        pairs = _transitions(seq)
        for pair in set(pairs):
            taken[pair] += 1
        for src in cond:
            # Self-loops are the loop's own rounds (see constant_rounds), not a dispatch order.
            orders[src].append(tuple(d for s, d in pairs if s == src and d != src))

    findings, times = [], f"{n} run{'s' * (n != 1)} of it"
    ran = {node for seq in seqs.values() for _, node in seq}
    for src, targets in sorted(cond.items()):
        if src not in ran:
            continue  # it never ran, so it never had a branch to take; the node that skipped it is the finding
        counts = {t: taken[(src, t)] for t in sorted(targets)}
        never = [t for t, c in counts.items() if c == 0]
        always = [t for t, c in counts.items() if c == n]
        if not targets:
            # Observed on langgraph 1.2.11: a branch function with no listed targets and no Literal
            # annotation is drawn with one edge, to the end. Where it can go is then not in the graph.
            went = sorted({d for s_, d in taken if s_ == src and d != src})
            findings.append({
                "kind": "targets_not_declared", "node": show(src), "runs": n,
                "text": f"{show(src)}: a branch point whose targets are not declared, so a branch never taken cannot be "
                        f"counted: read its function. On these runs it went to {', '.join(show(t) for t in went) or 'the end'}"})
        if never:
            findings.append({
                "kind": "branch_never_taken", "node": show(src), "targets": [show(t) for t in never], "runs": n,
                "text": f"{show(src)}: {len(targets)} branches declared, never taken in {times}: "
                        f"{', '.join(show(t) for t in never)}"
                        + (f"; always goes to {', '.join(show(t) for t in always)}" if always else "")})
        # A dispatcher chooses among different targets. The same target over and over is an
        # agent loop going round (model -> tools -> model), which is rounds, not dispatch.
        if n > 1 and orders[src] and len(set(orders[src])) == 1 and len(set(orders[src][0])) > 1:
            fixed, with_model = orders[src][0], src in llm_nodes
            findings.append({
                "kind": "fixed_dispatch", "node": show(src), "order": [show(t) for t in fixed], "runs": n,
                "calls_model": with_model,
                "text": f"{show(src)}: dispatches {' -> '.join(show(t) for t in fixed)} in the same order in {n}/{n} runs"
                        + (" while calling a model at every step to decide" if with_model else "")})

    rounds = defaultdict(list)
    for seq in seqs.values():
        for node, c in Counter(node for _, node in seq).items():
            rounds[node].append(c)
    for node, counts in sorted(rounds.items()):
        if n > 1 and len(counts) == n and len(set(counts)) == 1 and counts[0] > 1:
            findings.append({
                "kind": "constant_rounds", "node": show(node), "rounds": counts[0], "runs": n,
                "text": f"{show(node)}: ran exactly {counts[0]} times in every one of {n} runs"
                        " (a cap it always reaches, or a fixed schedule; check which)"})

    for node in sorted(llm_nodes):
        per_trace = [per[node] for per in replies.values() if len(per.get(node, [])) > 1]
        if len(replies) > 1 and len(per_trace) == len(replies) and all(len(set(r)) == 1 for r in per_trace):
            findings.append({
                "kind": "repeated_identical_reply", "node": show(node), "runs": len(replies),
                "text": f"{show(node)}: the model's reply is identical across all its rounds in "
                        f"{len(replies)}/{len(replies)} requests"})

    # Model calls spent per round of tool use. A tool-calling agent spends one per round, plus
    # one to give the answer: the same call picks the tool, writes its arguments and decides
    # whether to stop. Counted per request, over requests that used a tool, leaving out one
    # call each for the final answer (the first version divided all calls by all rounds and
    # called a plain two-node agent over-built: requests that needed no tool inflated it).
    llm, tool_rounds = Counter(), defaultdict(set)
    for r in runs:
        if here(r) and r["run_type"] == "llm":
            llm[r["trace_id"]] += 1
        elif here(r) and r["run_type"] == "tool":
            tool_rounds[r["trace_id"]].add(r["step"])
    calls, total = sum(max(llm[t] - 1, 0) for t in tool_rounds), sum(len(s) for s in tool_rounds.values())
    if total and calls / total >= 2:
        where = prefix.rstrip(":") or "the graph"
        findings.append({
            "kind": "calls_per_tool_round", "node": where, "model_calls": calls, "tool_rounds": total,
            "ratio": round(calls / total, 1),
            "text": f"{where}: {calls} model calls for {total} rounds of tool use in {len(tool_rounds)} requests, not "
                    f"counting each request's final answer ({calls / total:.1f} per round), spread over "
                    f"{', '.join(sorted(show(x) for x in llm_nodes))}; a tool-calling agent spends one per round"})
    return n, findings


def failures(runs, n):
    """Nodes where something raised, and nodes that paused for a human. An interrupt is
    LangGraph's human-in-the-loop mechanism, not a failure. An error on a call inside a node
    whose own run did not fail was caught there: the graph went on without that result, which
    is worth more attention than a crash."""
    raised = defaultdict(lambda: {"traces": set(), "message": "", "node_failed": False})
    for r in runs:
        if not r["error"]:
            continue
        cls = r["error"].split("(", 1)[0].strip() or "Error"
        kind = "interrupt" if cls == "GraphInterrupt" else "node_error"
        entry = raised[(kind, r["path"] or "(graph)", cls)]
        entry["traces"].add(r["trace_id"])
        entry["message"] = entry["message"] or " ".join(r["error"].split("Traceback", 1)[0].split())[:220]
        entry["node_failed"] |= r["run_type"] == "chain" and r["name"] in (r["node"], "LangGraph")
    out = []
    for (kind, node, cls), e in sorted(raised.items(), key=lambda kv: (kv[0][0] != "node_error", kv[0][1])):
        k = len(e["traces"])
        if kind == "interrupt":
            text = f"{node}: paused for a human in {k}/{n} requests"
        else:
            caught = "" if e["node_failed"] else " - caught inside the node, the graph continued without it"
            text = f"{node}: raised {cls} in {k}/{n} requests{caught}: {e['message']}"
        out.append({"kind": kind, "node": node, "error": cls, "count": k,
                    "swallowed": kind == "node_error" and not e["node_failed"], "text": text})
    return out


def analyze(conn, session_ids):
    if not session_ids:
        return {"requests": 0, "findings": []}
    # A recording made before paths were kept names a node by its own name only.
    runs = conn.execute(
        "SELECT run_id, parent_run_id, trace_id, name, run_type, node, COALESCE(path, node) AS path, step, completion, error"
        f"  FROM runs WHERE session_id IN {_in(session_ids)} ORDER BY start_time", session_ids).fetchall()
    requests = len({r["trace_id"] for r in runs if r["parent_run_id"] is None})
    nodes, edges = declared(conn, session_ids, {r["node"] for r in runs if r["node"]})
    findings, graphs = failures(runs, requests), []  # a node that raises comes first
    for prefix, (own, own_edges) in levels(nodes, edges).items():
        n, found = _level(prefix, own, own_edges, runs)
        if n:
            graphs.append({"graph": prefix.rstrip(":") or "the graph", "runs": n,
                           "nodes": [x for x in own if not x.startswith("__")],
                           "branch_points": sorted({e["source"] for e in own_edges if e["conditional"]})})
            findings += found
    return {"requests": requests, "distinct_inputs": distinct_inputs(conn, session_ids), "graphs": graphs,
            "findings": findings}


def render(result):
    if not result["requests"]:
        return "no runs to analyze"
    k = result["distinct_inputs"]
    lines = [f"{result['requests']} requests recorded, over at most {k} distinct inputs"]
    if k < result["requests"]:
        lines.append(f"note: the same inputs were run again, so n/n counts show consistency, not variety; the "
                     f"evidence is {k} inputs wide")
    for g in result["graphs"]:
        lines.append(f"{g['graph']}: ran {g['runs']} times, {len(g['nodes'])} nodes, "
                     f"{len(g['branch_points'])} branch points")
    if not result["findings"]:
        return "\n".join(lines) + "\nnothing structural stands out on these runs"
    return "\n".join(lines) + "\n" + "\n".join(f"- {f['text']}" for f in result["findings"])
