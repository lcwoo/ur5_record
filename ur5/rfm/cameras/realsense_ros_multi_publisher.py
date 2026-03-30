import json
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def _load_camera_list(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"camera map must be a JSON list: {path}")
    for ent in data:
        if "name" not in ent or "device_id" not in ent:
            raise ValueError(f"camera map entry must include name/device_id: {ent}")
    # Deterministic order (so "Subscribing cams" / launch order is stable).
    return sorted(data, key=lambda e: str(e["name"]))


def _make_image_msg(
    *,
    stamp,
    frame_id: str,
    height: int,
    width: int,
    encoding: str,
    data: bytes,
    step: int,
) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(height)
    msg.width = int(width)
    msg.encoding = str(encoding)
    msg.is_bigendian = 0
    msg.step = int(step)
    msg.data = data
    return msg


@dataclass
class Args:
    camera_map_file: str = "/home/lcw/ur5_lerobot/camera_port_map.json"
    # Topic prefix. Publishes:
    #   {topic_prefix}/{name}/color/image_raw
    #   {topic_prefix}/{name}/depth/image_raw
    topic_prefix: str = "/rs"
    # With many cameras, conservative defaults are more reliable on typical USB topologies.
    width: int = 640
    height: int = 480
    fps: int = 15
    enable_depth: bool = False
    strict_profile: bool = False
    # Stagger camera pipeline starts to avoid simultaneous USB / firmware load spikes.
    start_stagger_ms: int = 300
    # If a camera fails to start, keep retrying this often instead of exiting the worker.
    start_retry_s: float = 2.0
    # Optional allowlist of camera names (e.g., "cam0,cam6,cam7") for debugging.
    only: str = ""
    # Special handling for problematic cameras (comma-separated names)
    special_cams: str = ""
    # Special format order for those cameras (comma-separated among: YUYV,MJPEG,RGB8,BGR8)
    special_format_order: str = "YUYV,MJPEG,RGB8"
    # Special wait_for_frames timeout in milliseconds
    special_timeout_ms: int = 8000
    # Special backoff sequence (seconds), e.g., "1,2,4,8"
    special_backoff_seq: str = "1,2,4,8"


class _CameraWorker:
    def __init__(
        self,
        *,
        node: Node,
        name: str,
        device_id: str,
        topic_prefix: str,
        width: int,
        height: int,
        fps: int,
        enable_depth: bool,
        strict_profile: bool,
        start_retry_s: float,
        special_set: Optional[set] = None,
        special_format_order: Optional[List[str]] = None,
        special_timeout_ms: Optional[int] = None,
        special_backoff_seq: Optional[List[float]] = None,
    ):
        self.node = node
        self.name = name
        self.device_id = device_id
        self.topic_prefix = topic_prefix.rstrip("/")

        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.enable_depth = bool(enable_depth)
        self.strict_profile = bool(strict_profile)
        self.start_retry_s = float(start_retry_s)
        self.start_delay_s: float = 0.0
        # Special handling
        self.is_special = bool(special_set) and (self.name in special_set)
        self.special_timeout_ms = int(special_timeout_ms or 8000)
        self.special_backoff_seq = list(special_backoff_seq or [1.0, 2.0, 4.0, 8.0])
        self._special_backoff_idx = 0
        self.special_format_order = list(special_format_order or [])

        self.pub_rgb = node.create_publisher(
            Image, f"{self.topic_prefix}/{self.name}/color/image_raw", 10
        )
        self.pub_depth = None
        if self.enable_depth:
            self.pub_depth = node.create_publisher(
                Image, f"{self.topic_prefix}/{self.name}/depth/image_raw", 10
            )

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        import pyrealsense2 as rs

        if self.start_delay_s > 0:
            time.sleep(float(self.start_delay_s))

        # Enumerate and log supported color profiles to understand available formats/modes
        try:
            ctx = rs.context()
            devs = list(ctx.query_devices())
            target = None
            for d in devs:
                try:
                    if d.get_info(rs.camera_info.serial_number) == str(self.device_id):
                        target = d
                        break
                except Exception:
                    continue
            if target is not None:
                supported = []
                for s in target.sensors:
                    try:
                        profiles = s.get_stream_profiles()
                    except Exception:
                        continue
                    for p in profiles:
                        try:
                            if p.stream_type() != rs.stream.color:
                                continue
                            vp = p.as_video_stream_profile()
                            fmt_enum = vp.format()
                            fmt_name = {
                                rs.format.yuyv: "YUYV",
                                rs.format.mjpeg: "MJPEG",
                                rs.format.rgb8: "RGB8",
                                rs.format.bgr8: "BGR8",
                            }.get(fmt_enum, str(fmt_enum))
                            w2 = vp.width()
                            h2 = vp.height()
                            fps2 = vp.fps()
                            supported.append((fmt_name, w2, h2, fps2))
                        except Exception:
                            continue
                # Deduplicate and sort for readability
                supported = sorted(list({(f, w2, h2, fps2) for (f, w2, h2, fps2) in supported}))
                # Compact summary per format
                by_fmt = {}
                for f, w2, h2, fps2 in supported:
                    by_fmt.setdefault(f, []).append(f"{w2}x{h2}@{fps2}")
                summary_parts = [f"{f}:[{', '.join(sorted(set(v)))}]" for f, v in by_fmt.items()]
                self.node.get_logger().info(
                    f"[rs_pub] {self.name} device_id={self.device_id} supported_color_profiles: " + " ".join(summary_parts)
                )
            else:
                self.node.get_logger().warn(f"[rs_pub] {self.name} device_id={self.device_id} not found during profile enumeration")
        except Exception as enum_e:
            self.node.get_logger().warn(f"[rs_pub] {self.name} failed to enumerate profiles: {enum_e}")

        # Map string names to librealsense formats
        def _fmt_name_to_val(fmt_name: str):
            m = {
                "YUYV": rs.format.yuyv,
                "MJPEG": rs.format.mjpeg,
                "RGB8": rs.format.rgb8,
                "BGR8": rs.format.bgr8,
            }
            return m.get(fmt_name.upper().strip())

        # Build preferred profiles:
        #   - Keep requested FPS (default 30), try lower resolutions first
        #   - If still failing, reduce FPS gradually
        req = (self.width, self.height, self.fps)
        if self.strict_profile:
            profiles = [req]
        else:
            target_fps = int(self.fps)
            res_candidates = [
                (self.width, self.height),
                (424, 240),
                (320, 240),
                (320, 180),
                (256, 144),
            ]
            _seen_res = set()
            res_candidates = [r for r in res_candidates if not (r in _seen_res or _seen_res.add(r))]
            profiles = []
            # Keep 30 (or target) fps first
            keep_fps = min(30, target_fps)
            for (rw, rh) in res_candidates:
                profiles.append((rw, rh, keep_fps))
            # Then lower fps if needed
            for f in [25, 20, 15, 10]:
                for (rw, rh) in res_candidates:
                    profiles.append((rw, rh, f))

        while not self._stop.is_set() and rclpy.ok():
            pipeline = None
            last_err: Optional[Exception] = None
            used = None
            current_profile_idx = -1
            used_format = None
            for (w, h, fps) in profiles:
                for _ in range(3):
                    try:
                        # Format order (default) or special override
                        if self.is_special and self.special_format_order:
                            fmt_order_list = []
                            for n in self.special_format_order:
                                v = _fmt_name_to_val(n)
                                if v is not None:
                                    fmt_order_list.append((n.upper(), v))
                        else:
                            # Default: YUYV -> MJPEG -> RGB8
                            fmt_order_list = [
                                ("YUYV", rs.format.yuyv),
                                ("MJPEG", rs.format.mjpeg),
                                ("RGB8", rs.format.rgb8),
                            ]
                        fmt_order: List[Tuple[str, int]] = fmt_order_list
                        last_fmt_err: Optional[Exception] = None
                        for fmt_name, fmt_val in fmt_order:
                            try:
                                pipeline = rs.pipeline()
                                config = rs.config()
                                config.enable_device(str(self.device_id))
                                config.enable_stream(rs.stream.color, w, h, fmt_val, fps)
                                if self.enable_depth:
                                    config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
                                start_ts = time.time()
                                pipeline.start(config)
                                used = (w, h, fps)
                                used_format = fmt_name
                                last_err = None
                                current_profile_idx = profiles.index(used)
                                self.node.get_logger().info(
                                    f"[rs_pub] {self.name} pipeline_started ts={start_ts:.3f} format={used_format} profile={w}x{h}@{fps}"
                                )
                                break
                            except Exception as fe:
                                self.node.get_logger().warn(
                                    f"[rs_pub] {self.name} start failed format={fmt_name} profile={w}x{h}@{fps}: {fe}"
                                )
                                last_fmt_err = fe
                                try:
                                    pipeline.stop()
                                except Exception:
                                    pass
                                pipeline = None
                                continue
                        if used is not None and used_format is not None:
                            break
                        else:
                            last_err = last_fmt_err
                            time.sleep(0.2)
                    except Exception as e:
                        last_err = e
                        time.sleep(0.3)
                if used is not None:
                    break

            if pipeline is None or used is None:
                self.node.get_logger().warn(
                    f"RealSense start failed for name={self.name} device_id={self.device_id} "
                    f"profile={req} enable_depth={self.enable_depth} strict={self.strict_profile}: {last_err} "
                    f"(retry in {self.start_retry_s:.1f}s)"
                )
                time.sleep(max(0.2, self.start_retry_s))
                continue

            if used != req:
                self.node.get_logger().warn(
                    f"[rs_pub] {self.name} device_id={self.device_id} requested {req[0]}x{req[1]}@{req[2]} "
                    f"-> using {used[0]}x{used[1]}@{used[2]} format={used_format}"
                )
            else:
                self.node.get_logger().info(
                    f"[rs_pub] {self.name} device_id={self.device_id} started {used[0]}x{used[1]}@{used[2]} "
                    f"format={used_format} depth={self.enable_depth}"
                )

            w, h, _ = used
            frame_id = f"rs_{self.name}"

            # Runtime robustness: under heavy USB load, frames may time out. Don't crash the thread.
            # We'll keep trying and periodically restart the pipeline if errors persist.
            err_count = 0
            last_err_log_ts = 0.0
            err_log_period_s = 5.0
            restart_after = 10  # consecutive failures before restart
            # Achieved FPS measurement per camera
            _fps_count = 0
            _fps_last_log_ts = time.time()
            _fps_log_period_s = 5.0
            # Once camera achieves near-target FPS consistently, stop periodic INFO logs
            _fps_locked = False
            _fps_lock_hits = 0
            _fps_lock_required_hits = 2  # require 2 consecutive good windows before locking
            _fps_lock_tolerance = 0.95   # >= 95% of requested fps counts as good
            # After locking, avoid adaptive profile changes; only restart same profile on errors
            _adaptive_locked = False
            # First frame latency tracking and timeout counter
            _first_frame_ts: Optional[float] = None
            _timeout_count: int = 0

            try:
                while not self._stop.is_set() and rclpy.ok():
                    try:
                        wait_ms = self.special_timeout_ms if self.is_special else 5000
                        frames = pipeline.wait_for_frames(wait_ms)
                    except Exception as e:
                        err_count += 1
                        _timeout_count += 1
                        now_ts = time.time()
                        if now_ts - last_err_log_ts >= err_log_period_s:
                            self.node.get_logger().warn(
                                f"[rs_pub] {self.name} device_id={self.device_id} "
                                f"wait_for_frames timeout/error (x{err_count}, total_timeouts={_timeout_count}): {e}"
                            )
                            last_err_log_ts = now_ts

                        if err_count >= restart_after:
                            try:
                                pipeline.stop()
                            except Exception:
                                pass
                            # Backoff before attempting restart/switch
                            if self.is_special and self.special_backoff_seq:
                                backoff_s = self.special_backoff_seq[min(self._special_backoff_idx, len(self.special_backoff_seq) - 1)]
                                self._special_backoff_idx = min(self._special_backoff_idx + 1, len(self.special_backoff_seq) - 1)
                                self.node.get_logger().warn(f"[rs_pub] {self.name} applying backoff {backoff_s:.1f}s before restart/switch")
                                time.sleep(float(backoff_s))
                            else:
                                time.sleep(0.5)
                            if not _adaptive_locked:
                                # Attempt adaptive fallback to the next conservative profile
                                next_idx = min(current_profile_idx + 1, len(profiles) - 1)
                                next_profile = profiles[next_idx]
                                self.node.get_logger().warn(
                                    f"[rs_pub] {self.name} device_id={self.device_id} switching profile "
                                    f"{used[0]}x{used[1]}@{used[2]} -> {next_profile[0]}x{next_profile[1]}@{next_profile[2]} "
                                    f"after {err_count} errors"
                                )
                                try:
                                    # Try next profile with preferred format order again
                                    if self.is_special and self.special_format_order:
                                        fmt_order_list = []
                                        for n in self.special_format_order:
                                            v = _fmt_name_to_val(n)
                                            if v is not None:
                                                fmt_order_list.append((n.upper(), v))
                                    else:
                                        fmt_order_list = [
                                            ("YUYV", rs.format.yuyv),
                                            ("MJPEG", rs.format.mjpeg),
                                            ("RGB8", rs.format.rgb8),
                                        ]
                                    fmt_order: List[Tuple[str, int]] = fmt_order_list
                                    last_fmt_err: Optional[Exception] = None
                                    started = False
                                    for fmt_name, fmt_val in fmt_order:
                                        try:
                                            pipeline = rs.pipeline()
                                            config = rs.config()
                                            config.enable_device(str(self.device_id))
                                            config.enable_stream(rs.stream.color, next_profile[0], next_profile[1], fmt_val, int(next_profile[2]))
                                            if self.enable_depth:
                                                config.enable_stream(rs.stream.depth, next_profile[0], next_profile[1], rs.format.z16, int(next_profile[2]))
                                            start_ts2 = time.time()
                                            pipeline.start(config)
                                            used = next_profile
                                            used_format = fmt_name
                                            w, h, _ = used
                                            current_profile_idx = next_idx
                                            err_count = 0
                                            self.node.get_logger().info(
                                                f"[rs_pub] {self.name} pipeline_restarted ts={start_ts2:.3f} format={used_format} profile={w}x{h}@{used[2]}"
                                            )
                                            started = True
                                            # Reset first-frame tracker on restart
                                            _first_frame_ts = None
                                            break
                                        except Exception as fe:
                                            last_fmt_err = fe
                                            try:
                                                pipeline.stop()
                                            except Exception:
                                                pass
                                            pipeline = None
                                            continue
                                    if not started:
                                        raise RuntimeError(f"Failed to start next profile with any format: {last_fmt_err}")
                                except Exception as e2:
                                    # Keep trying later if profile switch fails
                                    err_count = 0
                                    self.node.get_logger().error(
                                        f"[rs_pub] {self.name} device_id={self.device_id} profile switch failed: {e2}"
                                    )
                                    time.sleep(1.0)
                            else:
                                # Adaptive locked: restart same profile only
                                try:
                                    if self.is_special and self.special_format_order:
                                        fmt_order_list = []
                                        for n in self.special_format_order:
                                            v = _fmt_name_to_val(n)
                                            if v is not None:
                                                fmt_order_list.append((n.upper(), v))
                                    else:
                                        fmt_order_list = [
                                            ("YUYV", rs.format.yuyv),
                                            ("MJPEG", rs.format.mjpeg),
                                            ("RGB8", rs.format.rgb8),
                                        ]
                                    fmt_order: List[Tuple[str, int]] = fmt_order_list
                                    restarted = False
                                    for fmt_name, fmt_val in fmt_order:
                                        try:
                                            pipeline = rs.pipeline()
                                            config = rs.config()
                                            config.enable_device(str(self.device_id))
                                            config.enable_stream(rs.stream.color, w, h, fmt_val, int(used[2]))
                                            if self.enable_depth:
                                                config.enable_stream(rs.stream.depth, w, h, rs.format.z16, int(used[2]))
                                            start_ts3 = time.time()
                                            pipeline.start(config)
                                            used_format = fmt_name
                                            err_count = 0
                                            self.node.get_logger().info(
                                                f"[rs_pub] {self.name} pipeline_restarted_same ts={start_ts3:.3f} format={used_format} profile={w}x{h}@{used[2]}"
                                            )
                                            _first_frame_ts = None
                                            restarted = True
                                            break
                                        except Exception:
                                            try:
                                                pipeline.stop()
                                            except Exception:
                                                pass
                                            pipeline = None
                                            continue
                                    if not restarted:
                                        self.node.get_logger().error(
                                            f"[rs_pub] {self.name} device_id={self.device_id} failed to restart locked profile {w}x{h}@{used[2]}"
                                        )
                                        time.sleep(1.0)
                                except Exception as e3:
                                    err_count = 0
                                    self.node.get_logger().error(
                                        f"[rs_pub] {self.name} device_id={self.device_id} same-profile restart failed: {e3}"
                                    )
                        continue

                    now = self.node.get_clock().now().to_msg()

                    c = frames.get_color_frame()
                    if c:
                        if _first_frame_ts is None:
                            _first_frame_ts = time.time()
                            self.node.get_logger().info(
                                f"[rs_pub] {self.name} first_frame_latency_ms={( (_first_frame_ts - (start_ts if 'start_ts' in locals() else time.time())) * 1000.0):.1f} "
                                f"format={used_format} timeouts={_timeout_count}"
                            )
                        # Publish raw buffers directly to minimize extra copies
                        if used_format == "YUYV":
                            encoding = "yuv422"
                            step_val = w * 2
                            data_bytes = memoryview(c.get_data()).tobytes()
                        elif used_format == "RGB8":
                            encoding = "rgb8"
                            step_val = w * 3
                            data_bytes = memoryview(c.get_data()).tobytes()
                        else:
                            # BGR8
                            if used_format == "MJPEG":
                                jpeg = np.frombuffer(memoryview(c.get_data()), dtype=np.uint8)
                                bgr = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
                                if bgr is None:
                                    raise RuntimeError(f"{self.name}: MJPEG decode failed")
                                encoding = "bgr8"
                                step_val = bgr.shape[1] * 3
                                data_bytes = bgr.tobytes()
                            else:
                                encoding = "bgr8"
                                step_val = w * 3
                                data_bytes = memoryview(c.get_data()).tobytes()
                        msg = _make_image_msg(
                            stamp=now,
                            frame_id=frame_id,
                            height=h,
                            width=w,
                            encoding=encoding,
                            data=data_bytes,
                            step=step_val,
                        )
                        self.pub_rgb.publish(msg)
                        err_count = 0
                        # FPS accounting
                        _fps_count += 1
                        _now_ts = time.time()
                        if _now_ts - _fps_last_log_ts >= _fps_log_period_s:
                            elapsed = max(1e-3, _now_ts - _fps_last_log_ts)
                            achieved_fps = _fps_count / elapsed
                            # Lock logic: once stabilized near target, emit one-time lock message and suppress further INFO
                            if not _fps_locked:
                                target = max(1.0, float(self.fps))
                                if achieved_fps >= (target * _fps_lock_tolerance):
                                    _fps_lock_hits += 1
                                else:
                                    _fps_lock_hits = 0

                                if _fps_lock_hits >= _fps_lock_required_hits:
                                    _fps_locked = True
                                    _adaptive_locked = True
                                    self.node.get_logger().info(
                                        f"[rs_pub] {self.name} locked_fps={achieved_fps:.1f} at {w}x{h}@{used[2]} format={used_format} depth={self.enable_depth}"
                                    )
                                else:
                                    self.node.get_logger().info(
                                        f"[rs_pub] {self.name} achieved_fps={achieved_fps:.1f} at {w}x{h}@{used[2]} format={used_format} depth={self.enable_depth}"
                                    )
                            # If locked, suppress periodic INFO logs; WARN/ERROR logs elsewhere still appear
                            _fps_last_log_ts = _now_ts
                            _fps_count = 0

                    if self.enable_depth and self.pub_depth is not None:
                        d = frames.get_depth_frame()
                        if d:
                            msg = _make_image_msg(
                                stamp=now,
                                frame_id=frame_id,
                                height=h,
                                width=w,
                                encoding="16UC1",
                                data=memoryview(d.get_data()).tobytes(),
                                step=w * 2,
                            )
                            self.pub_depth.publish(msg)
            finally:
                try:
                    pipeline.stop()
                except Exception:
                    pass


class RealSenseMultiPublisher(Node):
    def __init__(self, args: Args):
        super().__init__("ur5_realsense_multi_publisher")
        self.args = args
        cams = _load_camera_list(args.camera_map_file)
        self.get_logger().info(f"Loaded camera map: {args.camera_map_file} (n={len(cams)})")

        only = [s.strip() for s in str(args.only).split(",") if s.strip()]
        if only:
            cams = [ent for ent in cams if str(ent["name"]) in set(only)]
            self.get_logger().info(f"Filtering cameras (--only): {only} -> n={len(cams)}")

        # Parse special cams/options
        special_set = set([s.strip() for s in str(args.special_cams).split(",") if s.strip()])
        special_format_order = [s.strip() for s in str(args.special_format_order).split(",") if s.strip()]
        try:
            special_backoff_seq = [float(x) for x in str(args.special_backoff_seq).split(",") if x.strip()]
        except Exception:
            special_backoff_seq = [1.0, 2.0, 4.0, 8.0]

        self._workers: List[_CameraWorker] = []
        for idx, ent in enumerate(cams):
            w = _CameraWorker(
                node=self,
                name=str(ent["name"]),
                device_id=str(ent["device_id"]),
                topic_prefix=args.topic_prefix,
                width=args.width,
                height=args.height,
                fps=args.fps,
                enable_depth=args.enable_depth,
                strict_profile=args.strict_profile,
                start_retry_s=args.start_retry_s,
                special_set=special_set,
                special_format_order=special_format_order,
                special_timeout_ms=args.special_timeout_ms,
                special_backoff_seq=special_backoff_seq,
            )
            w.start_delay_s = float(idx) * (float(args.start_stagger_ms) / 1000.0)
            self._workers.append(w)

        for w in self._workers:
            w.start()


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--camera-map-file", default=Args.camera_map_file)
    p.add_argument("--topic-prefix", default=Args.topic_prefix)
    p.add_argument("--width", type=int, default=Args.width)
    p.add_argument("--height", type=int, default=Args.height)
    p.add_argument("--fps", type=int, default=Args.fps)
    p.add_argument("--start-stagger-ms", type=int, default=Args.start_stagger_ms)
    p.add_argument("--start-retry-s", type=float, default=Args.start_retry_s)
    p.add_argument("--only", default=Args.only)
    p.add_argument("--enable-depth", action="store_true", default=Args.enable_depth)
    p.add_argument("--no-depth", action="store_true", default=False)
    p.add_argument("--strict-profile", action="store_true", default=Args.strict_profile)
    p.add_argument("--allow-fallback", action="store_true", default=False)
    # Special handling CLI
    p.add_argument("--special-cams", default=Args.special_cams)
    p.add_argument("--special-format-order", default=Args.special_format_order)
    p.add_argument("--special-timeout-ms", type=int, default=Args.special_timeout_ms)
    p.add_argument("--special-backoff-seq", default=Args.special_backoff_seq)
    args_ns = p.parse_args()

    enable_depth = bool(args_ns.enable_depth) and (not bool(args_ns.no_depth))
    strict = bool(args_ns.strict_profile) and (not bool(args_ns.allow_fallback))

    args = Args(
        camera_map_file=str(args_ns.camera_map_file),
        topic_prefix=str(args_ns.topic_prefix),
        width=int(args_ns.width),
        height=int(args_ns.height),
        fps=int(args_ns.fps),
        enable_depth=enable_depth,
        strict_profile=strict,
        start_stagger_ms=int(args_ns.start_stagger_ms),
        start_retry_s=float(args_ns.start_retry_s),
        only=str(args_ns.only),
        special_cams=str(args_ns.special_cams),
        special_format_order=str(args_ns.special_format_order),
        special_timeout_ms=int(args_ns.special_timeout_ms),
        special_backoff_seq=str(args_ns.special_backoff_seq),
    )

    rclpy.init()
    node = RealSenseMultiPublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

