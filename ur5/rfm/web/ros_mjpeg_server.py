import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def _img_to_rgb_np(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    if msg.encoding == "rgb8":
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape((h, w, 3))
        return arr
    if msg.encoding == "bgr8":
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape((h, w, 3))
        return arr[:, :, ::-1]
    # Support YUYV (yuv422) directly for convenience
    if msg.encoding in ("yuv422", "yuyv", "yuyv422"):
        yuyv = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape((h, w, 2))
        # OpenCV expects interleaved YUYV (Y0 U0 Y1 V0 ...)
        bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
        return bgr[:, :, ::-1]  # to RGB
    raise ValueError(f"Unsupported encoding for MJPEG: {msg.encoding}")


class _FrameStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Dict[str, np.ndarray] = {}

    def set(self, name: str, rgb: np.ndarray) -> None:
        with self._lock:
            self._latest[name] = rgb

    def get(self, name: str) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest.get(name)

    def names(self):
        with self._lock:
            return list(self._latest.keys())


class RosImageCollector(Node):
    def __init__(self, camera_names, topic_prefix: str, store: _FrameStore):
        super().__init__("ur5_ros_mjpeg_collector")
        self.store = store
        prefix = topic_prefix.rstrip("/")
        for name in camera_names:
            topic = f"{prefix}/{name}/color/image_raw"
            self.create_subscription(Image, topic, lambda msg, n=name: self._on(n, msg), 10)
        self.get_logger().info(f"Subscribing RGB topics for cams: {list(camera_names)} (prefix={prefix})")

    def _on(self, name: str, msg: Image):
        try:
            rgb = _img_to_rgb_np(msg)
            self.store.set(name, rgb)
        except Exception:
            # Keep server alive even if one message is weird
            pass


def _load_camera_names(camera_map_file: str):
    data = json.loads(Path(camera_map_file).read_text())
    if not isinstance(data, list):
        raise ValueError("camera map must be a JSON list")
    # Deterministic order (stable HTML grid and subscriptions).
    return sorted([str(ent["name"]) for ent in data])


def _make_index_html(camera_names, host: str, port: int) -> str:
    cams = "".join(
        f"""
        <div class="cam">
          <div class="title">{name}</div>
          <img src="/mjpeg/{name}" />
        </div>
        """
        for name in camera_names
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>UR5 Camera Streams</title>
  <style>
    body {{ margin: 0; font-family: sans-serif; background: #111; color: #eee; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; padding: 8px; }}
    .cam {{ background: #1b1b1b; border-radius: 8px; overflow: hidden; }}
    .title {{ padding: 6px 8px; font-size: 14px; opacity: 0.9; }}
    img {{ width: 100%; height: auto; display: block; }}
    .hint {{ padding: 8px; font-size: 12px; opacity: 0.8; }}
  </style>
</head>
<body>
  <div class="hint">Open: http://{host}:{port}/ (MJPEG). If a tile stays black, that camera hasn't published a frame yet.</div>
  <div class="grid">{cams}</div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-map-file", default="/home/lcw/ur5_lerobot/camera_port_map.json")
    ap.add_argument("--topic-prefix", default="/rs")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--max-fps", type=float, default=15.0)
    args = ap.parse_args()

    camera_names = _load_camera_names(args.camera_map_file)
    store = _FrameStore()

    rclpy.init()
    node = RosImageCollector(camera_names, args.topic_prefix, store)

    # Spin ROS in background
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # JPEG encoder (OpenCV)
    import cv2

    boundary = "frame"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                html = _make_index_html(camera_names, self.headers.get("Host", "localhost").split(":")[0], args.port)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))
                return

            if self.path.startswith("/mjpeg/"):
                name = self.path.split("/mjpeg/", 1)[1].strip("/")
                if name not in camera_names:
                    self.send_response(404)
                    self.end_headers()
                    return

                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.end_headers()

                last_ts = 0.0
                min_period = 1.0 / max(float(args.max_fps), 1e-6)
                try:
                    while True:
                        now = time.time()
                        if now - last_ts < min_period:
                            time.sleep(0.001)
                            continue
                        last_ts = now

                        rgb = store.get(name)
                        if rgb is None:
                            # send a black placeholder until first frame arrives
                            rgb = np.zeros((240, 424, 3), dtype=np.uint8)

                        ok, jpg = cv2.imencode(
                            ".jpg",
                            rgb[:, :, ::-1],  # BGR
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
                        )
                        if not ok:
                            continue

                        self.wfile.write(f"--{boundary}\r\n".encode())
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg.tobytes())
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception:
                    return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args_):
            # quiet
            return

    server = ThreadingHTTPServer((args.bind, int(args.port)), Handler)
    try:
        print(f"[ur5-web] Serving MJPEG on http://{args.bind}:{args.port}/  (bind={args.bind})")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.shutdown()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

