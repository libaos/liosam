#!/usr/bin/env python3
from collections import Counter, deque
import threading
from typing import Optional

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from std_msgs.msg import Bool, Float32, Int32


class RouteIdGateNode:
    def __init__(self) -> None:
        self.input_id_topic = rospy.get_param("~input_id_topic", "/route_id")
        self.input_conf_topic = rospy.get_param("~input_conf_topic", "/route_conf")
        self.output_id_topic = rospy.get_param("~output_id_topic", "/route_id_stable")
        self.output_valid_topic = rospy.get_param("~output_valid_topic", "/route_id_valid")

        self.conf_th = float(rospy.get_param("~conf_th", 0.6))
        self.stable_n = max(1, int(rospy.get_param("~stable_N", 3)))
        self.timeout = float(rospy.get_param("~timeout", 2.0))
        self.allowed_jump = int(rospy.get_param("~allowed_jump", 1))
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))
        self.max_age = float(rospy.get_param("~max_age", 0.5))
        self.unknown_id = int(rospy.get_param("~unknown_id", -1))
        self.log_interval = float(rospy.get_param("~log_interval", 2.0))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.1))
        self.vote_window = max(0, int(rospy.get_param("~vote_window", 0)))
        self.vote_min_samples = max(1, int(rospy.get_param("~vote_min_samples", 3)))
        self.vote_switch_ratio = float(rospy.get_param("~vote_switch_ratio", 0.7))

        self._last_id: Optional[int] = None
        self._last_id_time: Optional[rospy.Time] = None
        self._last_conf: Optional[float] = None
        self._last_conf_time: Optional[rospy.Time] = None

        self._last_sample_id_time: Optional[rospy.Time] = None
        self._last_sample_conf_time: Optional[rospy.Time] = None

        self._stable_id: Optional[int] = None
        self._candidate_id: Optional[int] = None
        self._candidate_count = 0
        self._low_conf_start: Optional[rospy.Time] = None
        self._last_log_time = rospy.Time(0)
        self._id_history = deque(maxlen=self.vote_window if self.vote_window > 0 else 1)
        self._voted_id: Optional[int] = None

        self._id_pub = rospy.Publisher(self.output_id_topic, Int32, queue_size=1)
        self._valid_pub = rospy.Publisher(self.output_valid_topic, Bool, queue_size=1)

        self._id_sub = rospy.Subscriber(self.input_id_topic, Int32, self._id_callback, queue_size=1)
        self._conf_sub = rospy.Subscriber(self.input_conf_topic, Float32, self._conf_callback, queue_size=1)

        period = 1.0 / max(self.publish_rate, 1e-3)
        self._timer = rospy.Timer(rospy.Duration(period), self._on_timer)

        rospy.loginfo(
            "[route_id_gate] input_id=%s input_conf=%s output_id=%s",
            self.input_id_topic,
            self.input_conf_topic,
            self.output_id_topic,
        )

    def _id_callback(self, msg: Int32) -> None:
        self._last_id = int(msg.data)
        self._last_id_time = rospy.Time.now()

    def _conf_callback(self, msg: Float32) -> None:
        self._last_conf = float(msg.data)
        self._last_conf_time = rospy.Time.now()

    def _on_timer(self, _event) -> None:
        now = rospy.Time.now()
        state = "UNKNOWN"
        output_id = self.unknown_id
        valid = False

        if self._last_id is None or self._last_conf is None:
            self._publish(output_id, valid, state, now)
            return
        if self._last_id_time is None or self._last_conf_time is None:
            self._publish(output_id, valid, state, now)
            return

        if self.sync_slop > 0.0:
            dt = abs((self._last_id_time - self._last_conf_time).to_sec())
            if dt > self.sync_slop:
                if self._stable_id is not None and self.timeout > 0.0:
                    output_id = self._stable_id
                    valid = False
                    state = "STALE"
                self._publish(output_id, valid, state, now)
                return

        if self.max_age > 0.0:
            age_id = (now - self._last_id_time).to_sec()
            age_conf = (now - self._last_conf_time).to_sec()
            age = max(age_id, age_conf)
            if age > self.max_age:
                if self._stable_id is not None and self.timeout > 0.0 and age <= self.timeout:
                    output_id = self._stable_id
                    valid = False
                    state = "STALE"
                self._publish(output_id, valid, state, now)
                return

        sample_updated = (
            self._last_sample_id_time != self._last_id_time or self._last_sample_conf_time != self._last_conf_time
        )
        if sample_updated:
            self._last_sample_id_time = self._last_id_time
            self._last_sample_conf_time = self._last_conf_time

        route_id = int(self._last_id)
        conf = float(self._last_conf)

        if route_id < 0:
            self._candidate_id = None
            self._candidate_count = 0
            self._low_conf_start = None
            if sample_updated:
                self._id_history.clear()
                self._voted_id = None
            self._publish(output_id, valid, state, now, conf)
            return

        if sample_updated and self.vote_window > 0:
            self._id_history.append(route_id)

        if conf < self.conf_th:
            if self._low_conf_start is None:
                self._low_conf_start = now
            if sample_updated:
                self._candidate_id = None
                self._candidate_count = 0
            if self._stable_id is not None and (now - self._low_conf_start).to_sec() <= self.timeout:
                output_id = self._stable_id
                valid = False
                state = "UNSURE"
            elif self.vote_window > 0 and len(self._id_history) >= self.vote_min_samples:
                voted, count = Counter(self._id_history).most_common(1)[0]
                ratio = float(count) / float(len(self._id_history))
                if self._voted_id is None:
                    self._voted_id = int(voted)
                elif int(voted) != int(self._voted_id) and ratio >= self.vote_switch_ratio:
                    self._voted_id = int(voted)
                output_id = int(self._voted_id)
                valid = False
                state = "VOTE"
            else:
                output_id = self.unknown_id
                valid = False
                state = "UNKNOWN"
            self._publish(output_id, valid, state, now, conf)
            return

        self._low_conf_start = None

        if self._stable_id is not None and self.allowed_jump >= 0:
            if abs(route_id - self._stable_id) > self.allowed_jump:
                output_id = self._stable_id
                valid = True
                state = "STABLE"
                rospy.logwarn_throttle(
                    2.0,
                    "[route_id_gate] reject jump %d -> %d",
                    self._stable_id,
                    route_id,
                )
                self._publish(output_id, valid, state, now, conf)
                return

        if sample_updated:
            if self._candidate_id == route_id:
                self._candidate_count += 1
            else:
                self._candidate_id = route_id
                self._candidate_count = 1

        if self._stable_id is None:
            if self._candidate_count >= self.stable_n:
                self._stable_id = route_id
                output_id = self._stable_id
                valid = True
                state = "STABLE"
            else:
                output_id = route_id
                valid = False
                state = "ACQUIRE"
        else:
            if route_id == self._stable_id:
                output_id = self._stable_id
                valid = True
                state = "STABLE"
            elif self._candidate_count >= self.stable_n:
                self._stable_id = route_id
                output_id = self._stable_id
                valid = True
                state = "STABLE"
            else:
                output_id = self._stable_id
                valid = False
                state = "UNSURE"

        self._publish(output_id, valid, state, now, conf)

    def _publish(self, output_id: int, valid: bool, state: str, now: rospy.Time, conf: Optional[float] = None) -> None:
        self._id_pub.publish(Int32(data=int(output_id)))
        self._valid_pub.publish(Bool(data=bool(valid)))

        if (now - self._last_log_time).to_sec() >= self.log_interval:
            if conf is None:
                conf = float("nan")
            rospy.loginfo(
                "[route_id_gate] state=%s id=%s valid=%s conf=%.2f cand=%s n=%d",
                state,
                output_id,
                valid,
                conf,
                self._candidate_id,
                self._candidate_count,
            )
            self._last_log_time = now


if __name__ == "__main__":
    rospy.init_node("route_id_gate")
    RouteIdGateNode()
    rospy.spin()
