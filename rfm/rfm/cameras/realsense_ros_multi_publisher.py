import json
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
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
    camera_map_file: str = "/home/lcw/RFM_lerobot/camera_port_map.json"
    # Topic prefix. Publishes:
    #   {topic_prefix}/{name}/color/image_raw
    #   {topic_prefix}/{name}/depth/image_raw
    topic_prefix: str = "/rs"
    # With many cameras, conservative defaults are more reliable on typical USB topologies.
    width: int = 424
    height: int = 240
    fps: int = 6
    enable_depth: bool = True
    strict_profile: bool = True
    # Stagger camera pipeline starts to avoid simultaneous USB / firmware load spikes.
    start_stagger_ms: int = 300
    # If a camera fails to start, keep retrying this often instead of exiting the worker.
    start_retry_s: float = 2.0
    # Optional allowlist of camera names (e.g., "cam0,cam6,cam7") for debugging.
    only: str = ""


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

        # Try to start with strict profile (no fallback) by default.
        # If strict_profile=False, we try a few common fallbacks but never increase fps.
        req = (self.width, self.height, self.fps)
        if self.strict_profile:
            profiles = [req]
        else:
            f = self.fps
            profiles = [
                req,
                (self.width, self.height, 5),
                (640, 480, min(15, f)),
                (640, 480, min(30, f)),
                (640, 480, 5),
                (640, 360, min(15, f)),
                (640, 360, min(30, f)),
                (640, 360, 5),
                (424, 240, min(15, f)),
                (424, 240, min(30, f)),
                (424, 240, 5),
            ]
            seen = set()
            profiles = [p for p in profiles if not (p in seen or seen.add(p))]

        while not self._stop.is_set() and rclpy.ok():
            pipeline = None
            last_err: Optional[Exception] = None
            used = None
            for (w, h, fps) in profiles:
                for _ in range(3):
                    try:
                        pipeline = rs.pipeline()
                        config = rs.config()
                        config.enable_device(str(self.device_id))
                        # Use bgr8 (widely supported), then publish as rgb8 after conversion.
                        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
                        if self.enable_depth:
                            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
                        pipeline.start(config)
                        used = (w, h, fps)
                        last_err = None
                        break
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
                    f"-> using {used[0]}x{used[1]}@{used[2]}"
                )
            else:
                self.node.get_logger().info(
                    f"[rs_pub] {self.name} device_id={self.device_id} started {used[0]}x{used[1]}@{used[2]} "
                    f"depth={self.enable_depth}"
                )

            w, h, _ = used
            frame_id = f"rs_{self.name}"

            # Runtime robustness: under heavy USB load, frames may time out. Don't crash the thread.
            # We'll keep trying and periodically restart the pipeline if errors persist.
            err_count = 0
            last_err_log_ts = 0.0
            err_log_period_s = 5.0
            restart_after = 10  # consecutive failures before restart

            try:
                while not self._stop.is_set() and rclpy.ok():
                    try:
                        frames = pipeline.wait_for_frames(5000)
                    except Exception as e:
                        err_count += 1
                        now_ts = time.time()
                        if now_ts - last_err_log_ts >= err_log_period_s:
                            self.node.get_logger().warn(
                                f"[rs_pub] {self.name} device_id={self.device_id} "
                                f"wait_for_frames timeout/error (x{err_count}): {e}"
                            )
                            last_err_log_ts = now_ts

                        if err_count >= restart_after:
                            self.node.get_logger().warn(
                                f"[rs_pub] {self.name} device_id={self.device_id} restarting pipeline after {err_count} errors"
                            )
                            try:
                                pipeline.stop()
                            except Exception:
                                pass
                            time.sleep(0.5)
                            # Recreate pipeline with the same used profile
                            try:
                                pipeline = rs.pipeline()
                                config = rs.config()
                                config.enable_device(str(self.device_id))
                                config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, int(used[2]))
                                if self.enable_depth:
                                    config.enable_stream(rs.stream.depth, w, h, rs.format.z16, int(used[2]))
                                pipeline.start(config)
                                err_count = 0
                            except Exception as e2:
                                # If restart fails, keep trying.
                                err_count = 0
                                self.node.get_logger().error(
                                    f"[rs_pub] {self.name} device_id={self.device_id} pipeline restart failed: {e2}"
                                )
                                time.sleep(1.0)
                        continue

                    now = self.node.get_clock().now().to_msg()

                    c = frames.get_color_frame()
                    if c:
                        bgr = np.asanyarray(c.get_data())  # (H,W,3) uint8 BGR
                        rgb = bgr[:, :, ::-1]
                        msg = _make_image_msg(
                            stamp=now,
                            frame_id=frame_id,
                            height=h,
                            width=w,
                            encoding="rgb8",
                            data=rgb.tobytes(),
                            step=w * 3,
                        )
                        self.pub_rgb.publish(msg)
                        err_count = 0

                    if self.enable_depth and self.pub_depth is not None:
                        d = frames.get_depth_frame()
                        if d:
                            depth = np.asanyarray(d.get_data())  # (H,W) uint16
                            msg = _make_image_msg(
                                stamp=now,
                                frame_id=frame_id,
                                height=h,
                                width=w,
                                encoding="16UC1",
                                data=depth.tobytes(),
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
        super().__init__("rfm_realsense_multi_publisher")
        self.args = args
        cams = _load_camera_list(args.camera_map_file)
        self.get_logger().info(f"Loaded camera map: {args.camera_map_file} (n={len(cams)})")

        only = [s.strip() for s in str(args.only).split(",") if s.strip()]
        if only:
            cams = [ent for ent in cams if str(ent["name"]) in set(only)]
            self.get_logger().info(f"Filtering cameras (--only): {only} -> n={len(cams)}")

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

