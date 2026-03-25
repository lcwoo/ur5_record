from dataclasses import dataclass
from multiprocessing import Process
import traceback
import json
import socket
from pathlib import Path
import time

import tyro

from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.zmq_core.camera_node import ZMQServerCamera


@dataclass
class Args:
    # Bind address for the ZMQ camera servers.
    # Use 0.0.0.0 to listen on all interfaces (clients can connect via 127.0.0.1 or LAN IP).
    hostname: str = "0.0.0.0"
    start_port: int = 5000
    end_port: int = 6000
    max_cameras: int = 8
    camera_map_file: str = "/home/lcw/RFM_lerobot/camera_port_map.json"
    # Camera stream config (lower resolution/FPS can dramatically reduce latency with many cameras).
    # Default to a widely supported profile. You can lower these for speed, but
    # some models don't support all combos (auto-fallback will kick in).
    width: int = 640
    height: int = 360
    fps: int = 15
    no_depth: bool = False
    strict_profile: bool = False


def _is_port_free(host: str, port: int) -> bool:
    """Check if a TCP port is available for bind on this host."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        return True
    except OSError:
        return False


def launch_server(port: int, camera_id: int, args: Args):
    try:
        camera = RealSenseCamera(
            camera_id,
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            enable_depth=(not bool(args.no_depth)),
            strict_profile=bool(args.strict_profile),
        )
        server = ZMQServerCamera(camera, port=port, host=args.hostname)
        print(f"Starting camera server on port {port} (device_id={camera_id})")
        server.serve()
    except Exception as e:
        print(f"[camera_server_crash] device_id={camera_id} port={port} error={e}")
        print(traceback.format_exc())


def main(args):
    # Avoid hardware resetting devices during enumeration; it can cause transient disconnects.
    ids = get_device_ids(reset=False)
    if args.max_cameras is not None and args.max_cameras > 0:
        ids = ids[: int(args.max_cameras)]

    # Pick free ports in [start_port, end_port)
    ports = []
    for p in range(int(args.start_port), int(args.end_port)):
        if _is_port_free("0.0.0.0" if args.hostname in ("0.0.0.0", "127.0.0.1", "localhost") else args.hostname, p):
            ports.append(p)
            if len(ports) >= len(ids):
                break
    if len(ports) < len(ids):
        raise RuntimeError(
            f"Not enough free ports in [{args.start_port}, {args.end_port}) "
            f"for {len(ids)} cameras (found {len(ports)})."
        )

    # Write a deterministic mapping file (by enumeration order).
    map_path = Path(args.camera_map_file)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    mapping = []
    for i, (camera_id, port) in enumerate(zip(ids, ports)):
        mapping.append(
            {"name": f"cam{i}", "device_id": str(camera_id), "port": int(port)}
        )
    map_path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    print(f"Wrote camera port map: {map_path} (n={len(mapping)})")

    camera_servers = []
    for camera_id, camera_port in zip(ids, ports):
        # start a python process for each camera
        print(f"Launching camera {camera_id} on port {camera_port}")
        p = Process(target=launch_server, args=(camera_port, camera_id, args))
        # If the parent is killed, we generally want camera servers to go down too.
        p.daemon = True
        camera_servers.append(p)

    for server in camera_servers:
        server.start()

    # Health check: print which servers are alive after a short grace period.
    time.sleep(2.0)
    for i, server in enumerate(camera_servers):
        if not server.is_alive():
            print(f"[camera_server_dead] idx={i} pid={server.pid} exitcode={server.exitcode}")

    # Keep the parent process alive so Ctrl+C can cleanly tear down child servers.
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[launch_camera_nodes] Caught Ctrl+C, stopping camera servers...")
        for server in camera_servers:
            if server.is_alive():
                server.terminate()
        for server in camera_servers:
            try:
                server.join(timeout=2.0)
            except Exception:
                pass
        print("[launch_camera_nodes] Done.")


if __name__ == "__main__":
    main(tyro.cli(Args))
