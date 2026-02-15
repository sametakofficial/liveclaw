"""Cross-platform global hotkey support for LiveClaw.

Layered approach — tries methods in order until one works:

1. XDG Desktop Portal GlobalShortcuts (Wayland: Hyprland, Sway, GNOME, KDE)
   - No config editing needed, user gets a system dialog to bind the key
2. pynput (X11, macOS, Windows)
3. File trigger fallback (always works everywhere)
   - touch /tmp/liveclaw-record-toggle

Also provides cross-platform desktop notifications.
"""

import logging
import os
import platform
import subprocess
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

CTRL_FILE = "/tmp/liveclaw-record-toggle"


# ─── Notifications ─────────────────────────────────────────────────────────────


def notify(title: str, body: str) -> None:
    """Send a desktop notification (non-blocking, cross-platform)."""
    try:
        system = platform.system()
        if system == "Linux":
            subprocess.Popen(
                ["notify-send", "-u", "normal", "-t", "2000",
                 "-a", "LiveClaw", title, body],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        elif system == "Darwin":
            subprocess.Popen(
                ["osascript", "-e",
                 f'display notification "{body}" with title "{title}"'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        elif system == "Windows":
            # PowerShell toast notification
            ps = (
                f'Add-Type -AssemblyName System.Windows.Forms; '
                f'$n = New-Object System.Windows.Forms.NotifyIcon; '
                f'$n.Icon = [System.Drawing.SystemIcons]::Information; '
                f'$n.Visible = $true; '
                f'$n.ShowBalloonTip(2000, "{title}", "{body}", "Info")'
            )
            subprocess.Popen(
                ["powershell", "-Command", ps],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass


# ─── Hotkey Manager ────────────────────────────────────────────────────────────


class HotkeyManager:
    """Cross-platform global hotkey manager with automatic backend selection."""

    def __init__(self, callback: Callable, shortcut: str = "ctrl+shift+r"):
        self._callback = callback
        self._shortcut = shortcut
        self._running = False
        self._threads: list[threading.Thread] = []
        self._method = "none"

    def start(self) -> str:
        """Start listening. Returns the method name that was activated."""
        self._running = True

        # Always start file trigger (universal fallback)
        self._start_file_trigger()

        system = platform.system()

        if system == "Linux":
            session = os.environ.get("XDG_SESSION_TYPE", "").lower()
            wayland = session == "wayland" or os.environ.get("WAYLAND_DISPLAY")

            if wayland:
                # Try XDG Portal first (Wayland native)
                if self._try_xdg_portal():
                    self._method = "xdg-portal"
                    return self._method

            # Try pynput (X11 or Wayland with XWayland)
            if not wayland and self._try_pynput():
                self._method = "pynput"
                return self._method

        elif system in ("Darwin", "Windows"):
            if self._try_pynput():
                self._method = "pynput"
                return self._method

        # File trigger is always running as fallback
        self._method = "file-trigger"
        return self._method

    def stop(self) -> None:
        """Stop all listeners."""
        self._running = False

    # ─── XDG Desktop Portal GlobalShortcuts ────────────────────────────────

    def _try_xdg_portal(self) -> bool:
        """Register global shortcut via XDG Desktop Portal.

        Works on Hyprland, Sway, GNOME 44+, KDE 6+.
        The compositor shows a dialog letting the user pick the key combo.
        """
        try:
            import dbus
            from dbus.mainloop.glib import DBusGMainLoop
            from gi.repository import GLib
        except ImportError:
            logger.debug("dbus/gi not available for XDG Portal")
            return False

        def _portal_thread():
            try:
                DBusGMainLoop(set_as_default=True)
                bus = dbus.SessionBus()

                portal = bus.get_object(
                    "org.freedesktop.portal.Desktop",
                    "/org/freedesktop/portal/desktop",
                )
                shortcuts_iface = dbus.Interface(
                    portal, "org.freedesktop.portal.GlobalShortcuts"
                )

                # Create session
                session_opts = {
                    "handle_token": dbus.String("liveclaw_session"),
                    "session_handle_token": dbus.String("liveclaw_session"),
                }
                request_path = shortcuts_iface.CreateSession(session_opts)
                logger.debug(f"Portal session request: {request_path}")

                # Wait for session to be created
                time.sleep(0.5)

                # Session path
                sender = bus.get_unique_name().replace(".", "_").replace(":", "")
                session_path = f"/org/freedesktop/portal/desktop/session/{sender}/liveclaw_session"

                # List shortcuts we want to bind
                shortcut_list = dbus.Array([
                    dbus.Struct([
                        dbus.String("liveclaw-record"),
                        dbus.Dictionary({
                            "description": dbus.String("Toggle voice recording"),
                            "preferred_trigger": dbus.String(self._shortcut),
                        }, signature="sv"),
                    ], signature="sa{sv}"),
                ], signature="(sa{sv})")

                bind_opts = {
                    "handle_token": dbus.String("liveclaw_bind"),
                }

                shortcuts_iface.BindShortcuts(
                    dbus.ObjectPath(session_path),
                    shortcut_list,
                    "",  # parent_window
                    bind_opts,
                )

                logger.info("XDG Portal: shortcuts bound, waiting for activation...")

                # Listen for Activated signal
                def on_activated(session_handle, shortcut_id, timestamp, options):
                    if shortcut_id == "liveclaw-record":
                        logger.info("XDG Portal: shortcut activated")
                        self._callback()

                bus.add_signal_receiver(
                    on_activated,
                    signal_name="Activated",
                    dbus_interface="org.freedesktop.portal.GlobalShortcuts",
                    path="/org/freedesktop/portal/desktop",
                )

                # Run GLib main loop
                loop = GLib.MainLoop()
                while self._running:
                    ctx = loop.get_context()
                    ctx.iteration(True)

            except Exception as e:
                logger.warning(f"XDG Portal failed: {e}")

        t = threading.Thread(target=_portal_thread, daemon=True)
        t.start()
        self._threads.append(t)

        # Give it a moment to see if it crashes
        time.sleep(1.0)
        if t.is_alive():
            logger.info("XDG Portal GlobalShortcuts active")
            return True
        return False

    # ─── pynput (X11 / macOS / Windows) ────────────────────────────────────

    def _try_pynput(self) -> bool:
        """Register hotkey via pynput."""
        try:
            from pynput import keyboard

            # Convert shortcut to pynput format
            parts = self._shortcut.lower().split("+")
            pynput_parts = []
            for p in parts[:-1]:
                p = p.strip()
                if p in ("super", "meta", "win", "logo", "cmd"):
                    pynput_parts.append("<cmd>")
                elif p in ("ctrl", "control"):
                    pynput_parts.append("<ctrl>")
                elif p in ("shift",):
                    pynput_parts.append("<shift>")
                elif p in ("alt",):
                    pynput_parts.append("<alt>")
            pynput_parts.append(parts[-1].strip())
            combo = "+".join(pynput_parts)

            listener = keyboard.GlobalHotKeys({combo: self._callback})
            listener.start()
            self._threads.append(listener)
            logger.info(f"pynput hotkey active: {combo}")
            return True
        except Exception as e:
            logger.debug(f"pynput failed: {e}")
            return False

    # ─── File trigger (universal fallback) ─────────────────────────────────

    def _start_file_trigger(self) -> None:
        """Watch for trigger file — works on any OS, any WM."""
        try:
            os.unlink(CTRL_FILE)
        except OSError:
            pass

        def _watcher():
            while self._running:
                if os.path.exists(CTRL_FILE):
                    try:
                        os.unlink(CTRL_FILE)
                    except OSError:
                        pass
                    self._callback()
                time.sleep(0.15)

        t = threading.Thread(target=_watcher, daemon=True)
        t.start()
        self._threads.append(t)
        logger.info(f"File trigger active: touch {CTRL_FILE}")
