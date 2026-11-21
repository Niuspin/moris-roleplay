"""CLI and application entrypoint for Moris Roleplay local web server."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

_APP_ROOT = str(Path(__file__).resolve().parent)
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from backend.config_store import ConfigError
from backend.llama_engine import EngineError
from backend.server import REPO_ROOT, build_application_server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description=(
            "Moris Roleplay — Lightweight local AI roleplay chat web application "
            "powered by markdown character skills and llama.cpp."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="HTTP server bind host (default: value from config.json or 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP server bind port 1..65535 (default: value from config.json or 7860).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "config.json"),
        help="Path to runtime configuration JSON file (auto-created if missing).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the web browser on startup.",
    )
    args = parser.parse_args(argv)
    if args.port is not None and not (1 <= args.port <= 65535):
        parser.error(f"argument --port: must be between 1 and 65535 (got {args.port})")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser()

    try:
        server = build_application_server(
            config_path=config_path,
            host=args.host,
            port=args.port,
            restore_skill=False,
        )
    except (ConfigError, EngineError, OSError) as exc:
        print(f"[Moris Roleplay] Failed to start server: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        print(
            f"[Moris Roleplay] Unexpected startup error ({type(exc).__name__}): {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    bind_host, bind_port = server.server_address[0], int(server.server_address[1])
    browser_host = "127.0.0.1" if bind_host in ("0.0.0.0", "") else bind_host
    url = f"http://{browser_host}:{bind_port}"

    shutdown_Initiated = threading.Event()

    def _trigger_shutdown(signum: int, frame: Any) -> None:
        if not shutdown_Initiated.is_set():
            shutdown_Initiated.set()
            threading.Thread(target=server.shutdown, name="moris-shutdown", daemon=True).start()

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _trigger_shutdown)
            except (ValueError, OSError):
                pass

    cfg = server.chat_service.get_config()
    should_open_browser = (not args.no_browser) and bool(
        cfg.get("server", {}).get("open_browser", True)
    )
    if should_open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()

    print(f"[Moris Roleplay] Server listening on {url}", flush=True)
    print(f"[Moris Roleplay] Config file: {config_path.resolve()}", flush=True)

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.chat_service.shutdown()
        except Exception:
            pass
        try:
            server.server_close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
