"""Sandbox compatibility patches for ROS Python tooling.

This repository is often executed in restricted/sandboxed environments where
introspecting network interfaces is not permitted. ROS (roslaunch/rosgraph)
uses `netifaces.interfaces()` to enumerate local addresses; if that syscall is
blocked, roslaunch fails before starting any nodes.

When this module is present on `PYTHONPATH`, Python will auto-import it via the
standard `site` mechanism (as `sitecustomize`), allowing us to apply a small,
targeted monkey-patch.
"""

from __future__ import annotations


def _patch_rosgraph_network() -> None:
    try:
        import rosgraph.network as network  # type: ignore
    except Exception:
        return

    original = getattr(network, "get_local_addresses", None)
    if not callable(original):
        return

    def get_local_addresses_patched():  # type: ignore[no-untyped-def]
        try:
            return original()
        except (PermissionError, OSError):
            # Fall back to loopback in sandboxes where netifaces is blocked.
            fallback = ["127.0.0.1"]
            try:
                network._local_addrs = fallback  # type: ignore[attr-defined]
            except Exception:
                pass
            return fallback

    network.get_local_addresses = get_local_addresses_patched  # type: ignore[assignment]


_patch_rosgraph_network()

