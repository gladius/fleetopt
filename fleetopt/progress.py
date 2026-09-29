"""A line every minute while a session works in silence, so nobody takes it for stuck."""

import asyncio
import contextlib
import time


@contextlib.asynccontextmanager
async def ticking(what, every=60):
    began = time.time()

    async def tick():
        while True:
            await asyncio.sleep(every)
            print(f"[fleetopt] still {what} ({int(time.time() - began) // 60} min)", flush=True)

    task = asyncio.create_task(tick())
    try:
        yield
    finally:
        task.cancel()
