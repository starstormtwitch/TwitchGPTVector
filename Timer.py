import asyncio
import threading, logging
from typing import Callable, Coroutine

logger = logging.getLogger(__name__)


class LoopingTimer(threading.Thread):
    """
    Thread that will continuously run `target(*args, **kwargs)`
    every `interval` seconds, until program termination.
    """
    def __init__(self, interval: int, target: Callable[..., Coroutine], *args, **kwargs) -> None:
        threading.Thread.__init__(self)
        self.interval = interval
        self.target = target
        self.args = args
        self.kwargs = kwargs

        self.stopped = threading.Event()
        self.daemon = True

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._run_async())

    async def _run_async(self):
        while not self.stopped.is_set():
            await asyncio.gather(self.target(*self.args, **self.kwargs))
            await asyncio.sleep(self.interval)

    def stop(self):
        """
        Stop the timer.
        """
        self.stopped.set()

