import os
import time
from typing import List, Optional, Tuple

import numpy as np

from gello.cameras.camera import CameraDriver


def get_device_ids(reset: bool = False) -> List[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        if reset:
            # Hardware reset can temporarily disconnect devices. Only use when explicitly requested.
            dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    if reset:
        time.sleep(2)
    return device_ids


class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(
        self,
        device_id: Optional[str] = None,
        flip: bool = False,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        strict_profile: bool = False,
    ):
        import pyrealsense2 as rs

        self._device_id = device_id
        self._enable_depth = bool(enable_depth)

        # NOTE: requested profiles are not always supported across mixed RealSense models.
        # "Couldn't resolve requests" means the requested stream configuration cannot be satisfied.
        # We try the requested profile first. If strict_profile=True, we do NOT fall back.
        # If we do fall back, we never increase FPS above the requested FPS (to avoid extra load).
        req_w, req_h, req_f = int(width), int(height), int(fps)
        if strict_profile:
            profiles_to_try = [(req_w, req_h, req_f)]
        else:
            profiles_to_try = [
                (req_w, req_h, req_f),
                # Prefer same FPS first, then lower FPS. Never try higher than req_f.
                (640, 480, min(15, req_f)),
                (640, 480, min(30, req_f)),
                (640, 360, min(15, req_f)),
                (640, 360, min(30, req_f)),
                (848, 480, min(15, req_f)),
                (848, 480, min(30, req_f)),
                (424, 240, min(15, req_f)),
                (424, 240, min(30, req_f)),
            ]
            # Deduplicate while preserving order.
            seen = set()
            profiles_to_try = [p for p in profiles_to_try if not (p in seen or seen.add(p))]

        last_err: Optional[Exception] = None
        started = False
        for (w, h, f) in profiles_to_try:
            # Camera bring-up can be flaky when multiple devices start simultaneously.
            # Retry a few times per profile.
            for _ in range(5):
                try:
                    pipeline = rs.pipeline()
                    config = rs.config()
                    if device_id is not None:
                        config.enable_device(device_id)
                    if self._enable_depth:
                        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)
                    config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
                    pipeline.start(config)
                    self._pipeline = pipeline
                    started = True
                    last_err = None
                    if (w, h, f) != (int(width), int(height), int(fps)):
                        print(
                            f"[realsense_camera] device_id={device_id} "
                            f"requested {int(width)}x{int(height)}@{int(fps)} "
                            f"-> using {w}x{h}@{f} (supported fallback)"
                        )
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(0.3)
            if started:
                break

        if not started:
            if strict_profile:
                raise RuntimeError(
                    f"Couldn't resolve requests for device_id={device_id} "
                    f"with strict profile {req_w}x{req_h}@{req_f} (enable_depth={self._enable_depth}): {last_err}"
                )
            raise last_err if last_err is not None else RuntimeError("Failed to start RealSense pipeline")

        self._flip = flip
        # Cache last successfully read frame so transient dropouts don't stall the whole system.
        self._last_frame: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,  # farthest: float = 0.12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """
        import cv2

        # NOTE: librealsense can occasionally fail to deliver frames when many devices
        # are running (USB bandwidth / transient sync issues). If we raise here, the
        # ZMQ camera server dies and the client can hang waiting for a reply.
        #
        # So we retry a few times, and fall back to last good frame (or zeros).
        last_err: Optional[Exception] = None
        for _ in range(3):
            try:
                frames = self._pipeline.wait_for_frames(15000)  # timeout_ms
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame() if self._enable_depth else None
                if not color_frame or (self._enable_depth and not depth_frame):
                    last_err = RuntimeError("Missing color/depth frame")
                    time.sleep(0.05)
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                if self._enable_depth:
                    depth_image = np.asanyarray(depth_frame.get_data())  # type: ignore[union-attr]
                else:
                    # Placeholder depth (will be uint16 like real depth)
                    depth_image = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint16)

                if img_size is None:
                    image = color_image[:, :, ::-1]
                    depth = depth_image
                else:
                    image = cv2.resize(color_image, img_size)[:, :, ::-1]
                    depth = cv2.resize(depth_image, img_size)

                # rotate 180 degree's because everything is upside down in order to center the camera
                if self._flip:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                    depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
                else:
                    depth = depth[:, :, None]

                self._last_frame = (image, depth)
                return image, depth
            except Exception as e:
                last_err = e
                time.sleep(0.1)

        # Fallback: last known good frame, otherwise a black frame with correct shape.
        if self._last_frame is not None:
            return self._last_frame

        h, w = (480, 640) if img_size is None else (int(img_size[1]), int(img_size[0]))
        image = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w, 1), dtype=np.uint16)
        # Keep a small hint in logs (but don't spam).
        if last_err is not None:
            # Print once per instance by caching last_frame (still None here).
            print(f"[realsense_camera] warning: returning blank frame for device_id={self._device_id}: {last_err}")
        self._last_frame = (image, depth)
        return image, depth


def _debug_read(camera, save_datastream=False):
    import cv2

    cv2.namedWindow("image")
    cv2.namedWindow("depth")
    counter = 0
    if not os.path.exists("images"):
        os.makedirs("images")
    if save_datastream and not os.path.exists("stream"):
        os.makedirs("stream")
    while True:
        time.sleep(0.1)
        image, depth = camera.read()
        depth = np.concatenate([depth, depth, depth], axis=-1)
        key = cv2.waitKey(1)
        cv2.imshow("image", image[:, :, ::-1])
        cv2.imshow("depth", depth)
        if key == ord("s"):
            cv2.imwrite(f"images/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"images/depth_{counter}.png", depth)
        if save_datastream:
            cv2.imwrite(f"stream/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"stream/depth_{counter}.png", depth)
        counter += 1
        if key == 27:
            break


if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    rs = RealSenseCamera(flip=True, device_id=device_ids[0])
    im, depth = rs.read()
    _debug_read(rs, save_datastream=True)
