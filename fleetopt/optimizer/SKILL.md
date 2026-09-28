# Making the change for one finding

You are given one finding of a review of this agent, with the evidence it rests on.
Make the change it describes, in the source, and stop. fleetopt does everything else:
it commits your change, runs the agent, compares it with the original, judges the
answers, and keeps the change or undoes it.

## How to work

1. **Read before editing.** The finding, then the source it points to. Use
   `query_traces` or `graph_shape` if you need a number the finding does not give.
2. **One change, the smallest that does what the finding says.** No refactoring on the
   way, no second improvement you noticed, no new dependency.
3. **Keep what must survive.** The entry point other code calls and what it returns;
   the cap on rounds or spend; what happens when a tool fails; anything written or sent
   while it runs; points where a human approves. **Never remove what ends a loop.** If
   the agent loops until a model says it is done, keep a limit on rounds, and add one
   if a crash was all that ended it before. An agent that no longer crashes and never
   stops is worse than the one you were given.
4. **Check it loads.** `python -m py_compile <file>` on what you changed. Never run the
   agent, its tests or its eval suite: fleetopt runs it, under watch.
5. **If you were told why an earlier attempt failed,** fix that, and only that.
6. **If the change cannot be made** (the finding is wrong about the code, it needs a
   new dependency, it would remove something that must survive), change nothing.

## Reply

Your last message is one line and nothing else:

- `DONE: <what you changed, in under 15 words>`, or
- `CANNOT: <why, in under 25 words>`
