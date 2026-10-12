"""Command line entry point (`anygrep` and `python -m anygrep`)."""

import signal
import sys


def run() -> int:
    """Run the CLI. A Ctrl-C while the modules are still loading (the first
    fraction of a second) ends with "Interrupted." and status 130, as it does
    later, instead of a traceback."""
    interrupted = False

    def on_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt

    if signal.getsignal(signal.SIGINT) is signal.default_int_handler:
        signal.signal(signal.SIGINT, on_interrupt)
    try:
        from .cli import main

        return main()
    except KeyboardInterrupt:
        pass
    except Exception:
        # An extension module interrupted during its import can fail with
        # another error (numpy: "ImportError: initialization failed").
        if not interrupted:
            raise
    print("Interrupted.", file=sys.stderr)
    return 130


if __name__ == "__main__":
    raise SystemExit(run())
