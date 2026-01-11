#!/usr/bin/env python3
"""Switch TebLocalPlannerROS parameters based on /fsm/mode.

Typical usage:
- straight mode: faster forward, slower rotation
- left/right mode: slower forward, faster rotation

This node updates parameters via dynamic_reconfigure, so it works without
restarting move_base.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import rospy
from dynamic_reconfigure.client import Client as DynClient
from std_msgs.msg import String


def _normalize_mode(text: str) -> str:
    return (text or "").strip().lower()


def _pick_profile(mode: str, default_profile: str) -> str:
    if mode in ("left", "right", "turn_left", "turn_right", "turn"):
        return "turn"
    if mode in ("straight", "forward"):
        return "straight"
    return default_profile


class TebModeSwitcher:
    def __init__(self) -> None:
        self.mode_topic = str(rospy.get_param("~mode_topic", "/fsm/mode"))
        self.teb_server = str(rospy.get_param("~teb_server", "/move_base/TebLocalPlannerROS"))
        self.default_profile = str(rospy.get_param("~default_profile", "straight")).strip().lower() or "straight"
        self.apply_on_start = bool(rospy.get_param("~apply_on_start", True))
        self.min_update_period_s = float(rospy.get_param("~min_update_period_s", 0.2))

        profiles_param = rospy.get_param("~profiles", {})
        self.profiles: Dict[str, Dict[str, Any]] = {}
        if isinstance(profiles_param, dict):
            for name, params in profiles_param.items():
                if not isinstance(params, dict):
                    continue
                self.profiles[str(name).strip().lower()] = dict(params)

        self._client: Optional[DynClient] = None
        self._current_profile: Optional[str] = None
        self._last_update_time = rospy.Time(0)
        self._last_attempt_time = rospy.Time(0)

        self._sub = rospy.Subscriber(self.mode_topic, String, self._on_mode, queue_size=1)
        rospy.loginfo(
            "[orchard_teb_mode_switcher] Listening on %s, target teb server: %s",
            self.mode_topic,
            self.teb_server,
        )

        if self.apply_on_start:
            rospy.Timer(rospy.Duration(0.2), self._apply_default_once, oneshot=True)

    def _apply_default_once(self, _evt: rospy.timer.TimerEvent) -> None:
        self._apply_profile(self.default_profile, reason="startup")

    def _get_client(self) -> DynClient:
        if self._client is None:
            self._client = DynClient(self.teb_server, timeout=2.0)
        return self._client

    def _apply_profile(self, profile: str, reason: str) -> None:
        profile = str(profile).strip().lower()
        params = self.profiles.get(profile, None)
        if not params:
            rospy.logwarn_throttle(5.0, "[orchard_teb_mode_switcher] Missing profile '%s' params", profile)
            return

        now = rospy.Time.now()
        if self._current_profile == profile:
            return
        if self.min_update_period_s > 0.0 and (now - self._last_update_time).to_sec() < float(self.min_update_period_s):
            return
        if (now - self._last_attempt_time).to_sec() < 0.2:
            return
        self._last_attempt_time = now

        try:
            client = self._get_client()
            client.update_configuration(dict(params))
            self._current_profile = profile
            self._last_update_time = now
            rospy.loginfo(
                "[orchard_teb_mode_switcher] Applied profile '%s' (%s): %s",
                profile,
                reason,
                ", ".join(f"{k}={v}" for k, v in sorted(params.items())),
            )
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "[orchard_teb_mode_switcher] Failed to update %s via dynamic_reconfigure: %s",
                self.teb_server,
                exc,
            )
            self._client = None

    def _on_mode(self, msg: String) -> None:
        mode = _normalize_mode(getattr(msg, "data", ""))
        profile = _pick_profile(mode, default_profile=self.default_profile)
        self._apply_profile(profile, reason=f"mode={mode or '∅'}")


def main() -> None:
    rospy.init_node("orchard_teb_mode_switcher", anonymous=False)
    TebModeSwitcher()
    rospy.spin()


if __name__ == "__main__":
    main()

