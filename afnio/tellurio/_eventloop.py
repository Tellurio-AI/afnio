import asyncio
import threading


class EventLoopThread:
    """
    Runs an asyncio event loop in a dedicated background thread.

    This class enables synchronous code to execute asynchronous coroutines
    by submitting them to a persistent event loop running outside the main thread.
    It manages the lifecycle of the event loop and its thread, providing methods
    to run coroutines and to shut down the loop cleanly.
    """

    def __init__(self):
        # Create a new asyncio event loop (not the default one)
        self.loop = asyncio.new_event_loop()
        # Start the event loop in a dedicated daemon thread
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro):
        """
        Submit a coroutine to the background event loop and block until it completes.

        Args:
            coro: The coroutine to run.

        Returns:
            The result of the coroutine.
        """
        # Schedule the coroutine to run in the background event loop
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        # Block until the coroutine completes and return its result
        return future.result()

    def shutdown(self):
        """
        Stop the event loop and wait for the background thread to finish.
        """
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()


# Singleton instance of the background event loop thread
_event_loop_thread = EventLoopThread()


def run_in_background_loop(coro):
    """
    Submit a coroutine to the background event loop and return its result.
    """
    return _event_loop_thread.run(coro)
