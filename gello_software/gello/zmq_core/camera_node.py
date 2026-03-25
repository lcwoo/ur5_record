import pickle
import threading
import time
from typing import Optional, Tuple

import numpy as np
import zmq

from gello.cameras.camera import CameraDriver

DEFAULT_CAMERA_PORT = 5000


class ZMQClientCamera(CameraDriver):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(
        self,
        port: int = DEFAULT_CAMERA_PORT,
        host: str = "127.0.0.1",
        timeout_ms: int = 2000,
    ):
        self._host = host
        self._port = int(port)
        self._timeout_ms = int(timeout_ms)
        self._context = zmq.Context()
        self._socket = None
        self._connect()
        self._last_frame: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._err_count = 0

    def _connect(self) -> None:
        """(Re)create the REQ socket. Needed because REQ has a strict send/recv state machine."""
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
        s = self._context.socket(zmq.REQ)
        # Don't hang forever if a camera server is slow/dead.
        s.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        s.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        s.setsockopt(zmq.LINGER, 0)
        # Make REQ more tolerant: allow a new send even if a previous recv timed out.
        # This avoids "Operation cannot be accomplished in current state".
        try:
            s.setsockopt(zmq.REQ_RELAXED, 1)
            s.setsockopt(zmq.REQ_CORRELATE, 1)
        except Exception:
            # Some libzmq builds may not support these options; we'll still recover by reconnecting.
            pass
        s.connect(f"tcp://{self._host}:{self._port}")
        self._socket = s

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        # pack the image_size and send it to the server
        try:
            send_message = pickle.dumps(img_size)
            assert self._socket is not None
            self._socket.send(send_message)
            state_dict = pickle.loads(self._socket.recv())
            self._last_frame = state_dict
            self._err_count = 0
            return state_dict
        except Exception:
            # Any ZMQ error (timeout or REQ state error) — reset socket and fall back.
            self._err_count += 1
            self._connect()
            # Return last known frame if available, else a blank frame.
            if self._last_frame is not None:
                return self._last_frame
            h, w = (480, 640) if img_size is None else (int(img_size[1]), int(img_size[0]))
            image = np.zeros((h, w, 3), dtype=np.uint8)
            depth = np.zeros((h, w, 1), dtype=np.uint16)
            self._last_frame = (image, depth)
            return self._last_frame


class ZMQServerCamera:
    def __init__(
        self,
        camera: CameraDriver,
        port: int = DEFAULT_CAMERA_PORT,
        host: str = "127.0.0.1",
    ):
        self._camera = camera
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Camera Sever Binding to {addr}, Camera: {camera}"
        print(debug_message)
        # This server uses a REP socket and waits for client requests.
        # If no client connects yet, recv() will time out; that's normal.
        self._timeout_message = f"[camera_server_waiting] no client yet (Camera: {camera})"
        self._last_wait_log_ts = 0.0
        self._wait_log_period_s = 10.0
        self._socket.bind(addr)
        self._stop_event = threading.Event()
        # Cache most recent frame so requests are fast and don't block on wait_for_frames().
        self._latest: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._latest_lock = threading.Lock()
        self._last_grab_err: Optional[str] = None
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()

    def _grab_loop(self) -> None:
        """Continuously grab frames in the background and cache the latest."""
        # Warm up: allow camera pipelines to start delivering frames.
        while not self._stop_event.is_set():
            try:
                frame = self._camera.read(None)
                with self._latest_lock:
                    self._latest = frame
                self._last_grab_err = None
            except Exception as e:
                self._last_grab_err = str(e)
                time.sleep(0.05)

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                img_size = pickle.loads(message)
                with self._latest_lock:
                    cached = self._latest

                # If we don't have a frame yet, fall back to direct read (may block)
                # but should eventually populate cache.
                if cached is None:
                    cached = self._camera.read(img_size)
                else:
                    image, depth = cached
                    if img_size is not None:
                        # Resize on-demand on the server side so requests stay fast.
                        import cv2

                        image = cv2.resize(image[:, :, ::-1], img_size)[:, :, ::-1]
                        depth2d = depth[:, :, 0] if depth.ndim == 3 else depth
                        depth_rs = cv2.resize(depth2d, img_size)[:, :, None]
                        cached = (image, depth_rs)
                self._socket.send(pickle.dumps(cached))
            except zmq.Again:
                # No client request received within timeout. This is expected when
                # run_env.py (client) isn't running yet, so rate-limit the log.
                now = time.time()
                if now - self._last_wait_log_ts >= self._wait_log_period_s:
                    print(self._timeout_message)
                    self._last_wait_log_ts = now

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()
