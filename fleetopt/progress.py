"""A line every minute while a session works in silence, so nobody takes it for stuck."""

import asyncio
import contextlib
import time


@contextlib.asynccontextmanager
async def ticking(what, every=60, said_at=None):
    """`said_at`: when something was last printed, so a minute with its own lines gets no tick."""
    began = time.time()

    async def tick():
        while True:
            await asyncio.sleep(every)
            if not said_at or time.time() - said_at() >= every:
                print(f"  still {what} ({int(time.time() - began) // 60} min)", flush=True)

    task = asyncio.create_task(tick())
    try:
        yield
    finally:
        task.cancel()
