"""State that belongs to one run, in a process that may hold several.

A run's state used to be one module-level dict, so one process could hold one run: two
experts on the same project, or the three starts of a survey, had to go one after another.
A Scoped dict lives in the task that began the run and in every task and thread started from
it: the SDK runs each tool call and hook in a task spawned inside `query()`, `asyncio.to_thread`
and `gather` copy the context, so a tool always sees the run it belongs to and never another.
Reads and writes are a dict's; `fresh()` starts a new one for this task.
"""

import contextvars


class Scoped:
    def __init__(self, name):
        self._var = contextvars.ContextVar(name)

    def _d(self):
        try:
            return self._var.get()
        except LookupError:  # a thread of its own, or a test: a dict of its own, kept for later reads
            fresh = {}
            self._var.set(fresh)
            return fresh

    def fresh(self):
        self._var.set({})

    def __getitem__(self, key):
        return self._d()[key]

    def __setitem__(self, key, value):
        self._d()[key] = value

    def __contains__(self, key):
        return key in self._d()

    def __getattr__(self, name):  # get, setdefault, update, clear, items, pop, ...
        return getattr(self._d(), name)


if __name__ == "__main__":  # the check: two tasks, each its own dict, and a thread sees its task's
    import asyncio

    S = Scoped("s")

    async def run(name):
        S.fresh()
        S["who"] = name
        await asyncio.sleep(0.01)
        seen = await asyncio.to_thread(lambda: S["who"])
        assert seen == S["who"] == name, (seen, S.get("who"), name)
        return S["who"]

    async def both():
        return await asyncio.gather(run("a"), run("b"))

    assert asyncio.run(both()) == ["a", "b"]
    S.clear()
    S.update(x=1)
    assert S["x"] == 1 and "x" in S and S.get("y") is None
    print("ok")
