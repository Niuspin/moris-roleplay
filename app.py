"""CLI and standalone desktop application entrypoint for Moris Roleplay."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

# In pythonw.exe (windowless Windows mode), stdout/stderr can be None.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

_APP_ROOT = str(Path(__file__).resolve().parent)
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from backend.config_store import ConfigError
from backend.llama_engine import EngineError, probe_tcp_port_available
from backend.server import REPO_ROOT, build_application_server


def _find_available_port(host: str, preferred_port: int) -> int:
    """Return preferred_port if free, otherwise find a nearby open TCP port."""
    for candidate in range(preferred_port, min(65536, preferred_port + 50)):
        try:
            probe_tcp_port_available(host, candidate)
            return candidate
        except EngineError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_qt_desktop_window(url: str, *, profile_dir: Path, on_ready_shutdown: Any) -> bool:
    """Launch a self-contained PySide6 QtWebEngine desktop window (zero local browser dependency)."""
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QColor
        from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow
    except Exception:
        return False

    profile_dir.mkdir(parents=True, exist_ok=True)
    qt_app = QApplication.instance() or QApplication([sys.argv[0]])
    qt_app.setApplicationName("Moris Roleplay")
    qt_app.setQuitOnLastWindowClosed(True)

    storage_dir = str((profile_dir / "qt_storage").resolve())
    cache_dir = str((profile_dir / "qt_cache").resolve())
    profile = QWebEngineProfile("MorisRoleplay", qt_app)
    profile.setPersistentStoragePath(storage_dir)
    profile.setCachePath(cache_dir)

    def _on_download(download: Any) -> None:
        suggested = download.downloadFileName() or "moris-chat.md"
        save_path, _ = QFileDialog.getSaveFileName(
            None,
            "导出对话记录",
            suggested,
            "Markdown (*.md);;All Files (*)",
        )
        if save_path:
            target = Path(save_path)
            download.setDownloadDirectory(str(target.parent))
            download.setDownloadFileName(target.name)
            download.accept()
        else:
            download.cancel()

    profile.downloadRequested.connect(_on_download)

    class _MorisMainWindow(QMainWindow):
        def closeEvent(self, event: Any) -> None:
            on_ready_shutdown()
            super().closeEvent(event)

    window = _MorisMainWindow()
    window.setWindowTitle("Moris Roleplay — 本地沉浸式 AI 角色扮演剧场")
    window.resize(1280, 840)
    window.setMinimumSize(920, 620)
    window.setStyleSheet("background-color: #15110e;")

    page = QWebEnginePage(profile, window)
    page.setBackgroundColor(QColor("#15110e"))

    view = QWebEngineView(window)
    view.setPage(page)
    view.setUrl(QUrl(url))

    window.setCentralWidget(view)
    window.show()
    qt_app.exec()
    return True


def _run_pywebview_desktop_window(url: str, *, profile_dir: Path, on_ready_shutdown: Any) -> bool:
    """Secondary fallback using pywebview native desktop window (no browser app)."""
    try:
        import webview  # type: ignore[import-not-found]
    except Exception:
        return False

    profile_dir.mkdir(parents=True, exist_ok=True)
    win = webview.create_window(
        "Moris Roleplay — 本地沉浸式 AI 角色扮演剧场",
        url,
        width=1280,
        height=840,
        min_size=(920, 620),
        background_color="#15110e",
    )
    win.events.closed += lambda: on_ready_shutdown()
    webview.start(private_mode=False, storage_path=str((profile_dir / "webview").resolve()))
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description=(
            "Moris Roleplay — Lightweight local AI roleplay chat desktop application "
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
        help="Headless server mode: do not open the standalone desktop window.",
    )
    parser.add_argument(
        "--browser-tab",
        action="store_true",
        help="Open in a browser tab instead of the standalone desktop window.",
    )
    args = parser.parse_args(argv)
    if args.port is not None and not (1 <= args.port <= 65535):
        parser.error(f"argument --port: must be between 1 and 65535 (got {args.port})")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser()

    chosen_port = args.port
    if chosen_port is None and not args.no_browser and not args.browser_tab:
        # In GUI desktop mode, avoid failing if default 7860 is occupied by another app.
        try:
            probe_tcp_port_available(args.host or "127.0.0.1", 7860)
        except EngineError:
            chosen_port = _find_available_port(args.host or "127.0.0.1", 7861)

    try:
        server = build_application_server(
            config_path=config_path,
            host=args.host,
            port=chosen_port,
            restore_skill=True,
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

    shutdown_initiated = threading.Event()

    def _trigger_shutdown(signum: int = 0, frame: Any = None) -> None:
        if not shutdown_initiated.is_set():
            shutdown_initiated.set()
            threading.Thread(target=server.shutdown, name="moris-shutdown", daemon=True).start()

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _trigger_shutdown)
            except (ValueError, OSError):
                pass

    cfg = server.chat_service.get_config()
    should_open_ui = (not args.no_browser) and bool(
        cfg.get("server", {}).get("open_browser", True)
    )

    print(f"[Moris Roleplay] Server listening on {url}", flush=True)
    print(f"[Moris Roleplay] Config file: {config_path.resolve()}", flush=True)

    try:
        if should_open_ui and not args.browser_tab:
            server_thread = threading.Thread(
                target=server.serve_forever,
                kwargs={"poll_interval": 0.2},
                name="moris-http-server",
                daemon=True,
            )
            server_thread.start()
            profile_dir = config_path.resolve().parent / ".desktop_profile"
            launched = _run_qt_desktop_window(
                url,
                profile_dir=profile_dir,
                on_ready_shutdown=_trigger_shutdown,
            )
            if not launched:
                launched = _run_pywebview_desktop_window(
                    url,
                    profile_dir=profile_dir,
                    on_ready_shutdown=_trigger_shutdown,
                )
            if not launched:
                print(
                    "[Moris Roleplay] Standalone GUI runtime (PySide6/pywebview) unavailable; "
                    "server remains running headless.",
                    file=sys.stderr,
                    flush=True,
                )
                server_thread.join()
            else:
                _trigger_shutdown()
                server_thread.join(timeout=3.0)
        else:
            if should_open_ui and args.browser_tab:
                threading.Timer(0.25, lambda: webbrowser.open(url)).start()
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
