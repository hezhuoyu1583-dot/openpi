"""Decode LINGLOONG rosbag2 episodes used by the dataset export.

Copied from linglong_rawdata_spec/linglong_parser.py. Only the bag reader,
alignment, and JPEG export used by prepare_sft stay here. Plotting and video
export are not included.
"""
from __future__ import annotations

import bisect
import csv
import json
import sqlite3
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


TOPICS = {
    "/driver_pvt": "driver",
    "/end_pos": "end_pos",
    "/camera/head/image_raw/compressed": "head",
    "/camera/left/image_raw/compressed": "left",
    "/camera/right/image_raw/compressed": "right",
}


class CDRReader:
    """Small little-endian CDR reader for the primitive types used here."""

    def __init__(self, payload: bytes):
        if len(payload) < 4 or payload[:2] != b"\x00\x01":
            raise ValueError("unsupported CDR encapsulation (expected little-endian CDR)")
        self.data, self.pos = payload, 4

    def align(self, n: int) -> None:
        self.pos = (self.pos + n - 1) // n * n

    def _take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError(f"truncated CDR payload at offset {self.pos} (need {n} bytes)")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def u32(self) -> int:
        self.align(4)
        return struct.unpack("<I", self._take(4))[0]

    def i32(self) -> int:
        self.align(4)
        return struct.unpack("<i", self._take(4))[0]

    def f32(self) -> float:
        self.align(4)
        return struct.unpack("<f", self._take(4))[0]

    def string(self) -> str:
        n = self.u32()
        raw = self._take(n)
        if n and raw.endswith(b"\0"):
            return raw[:-1].decode("utf-8", errors="replace")
        return raw.decode("utf-8", errors="replace")

    def bytes(self) -> bytes:
        n = self.u32()
        return self._take(n)

    def finish(self) -> None:
        if self.pos != len(self.data):
            tail = self.data[self.pos :]
            if any(tail):
                raise ValueError(f"unexpected trailing CDR data ({len(tail)} bytes)")


def _header(reader: CDRReader) -> Tuple[int, str]:
    sec, nsec = reader.i32(), reader.u32()
    return sec * 1_000_000_000 + nsec, reader.string()


def decode_driver(payload: bytes) -> Dict[str, Any]:
    reader = CDRReader(payload)
    stamp, frame_id = _header(reader)
    limbs: List[List[Dict[str, float]]] = []
    for _ in range(reader.u32()):
        joints = []
        for _ in range(reader.u32()):
            vals = struct.unpack("<6f", reader._take(24))
            joints.append(dict(zip(("q", "q_exp", "qd", "qd_exp", "tau", "tau_exp"), vals)))
        limbs.append(joints)
    reader.finish()
    return {"timestamp_ns": stamp, "frame_id": frame_id, "limbs": limbs}


def decode_end_pos(payload: bytes) -> Dict[str, Any]:
    reader = CDRReader(payload)
    stamp, frame_id = _header(reader)
    names = ("ee_pose_l", "ee_pose_l_exp", "ee_pose_r", "ee_pose_r_exp", "ee_pose_waist", "ee_pose_waist_exp")
    poses = {name: [reader.f32() for _ in range(7)] for name in names}
    reader.finish()
    return {"timestamp_ns": stamp, "frame_id": frame_id, **poses}


def decode_compressed_image(payload: bytes) -> Dict[str, Any]:
    reader = CDRReader(payload)
    stamp, frame_id = _header(reader)
    fmt, data = reader.string(), reader.bytes()
    reader.finish()
    return {"timestamp_ns": stamp, "frame_id": frame_id, "format": fmt, "data": data}


def locate_db(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file() and path.suffix == ".db3":
        return path
    if path.is_dir() and (path / "bag_0.db3").exists():
        return path / "bag_0.db3"
    if path.is_dir() and (path / "bag" / "bag_0.db3").exists():
        return path / "bag" / "bag_0.db3"
    raise FileNotFoundError(f"cannot find bag_0.db3 below {path}")


def _reject_lfs_pointer(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(128)
    except OSError:
        return
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the real file; run `git lfs pull --include=\"data/**\"` first"
        )


def read_episode(path: Path | str) -> Dict[str, Any]:
    """Read all known topics from an episode directory or bag_0.db3."""
    requested = Path(path)
    db = locate_db(requested)
    metadata_path = db.parent.parent / "__loongdata_metadata.json"
    if metadata_path.exists():
        _reject_lfs_pointer(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {}
    _reject_lfs_pointer(db)
    out: Dict[str, Any] = {
        "path": str(requested),
        "db": str(db),
        "metadata": metadata,
        "driver": [],
        "end_pos": [],
        "head": [],
        "left": [],
        "right": [],
    }
    connection = sqlite3.connect(str(db))
    try:
        topic_ids = {
            name: topic_id
            for topic_id, name, *_ in connection.execute("SELECT id,name,type,serialization_format FROM topics")
        }
        query = "SELECT topic_id,timestamp,data FROM messages ORDER BY timestamp"
        for topic_id, record_ns, payload in connection.execute(query):
            name = next((topic for topic, stored_id in topic_ids.items() if stored_id == topic_id), None)
            if name not in TOPICS:
                continue
            kind = TOPICS[name]
            if kind == "driver":
                obj = decode_driver(payload)
            elif kind == "end_pos":
                obj = decode_end_pos(payload)
            else:
                obj = decode_compressed_image(payload)
            obj["record_timestamp_ns"] = record_ns
            out[kind].append(obj)
    finally:
        connection.close()
    return out


def flatten_joints(limbs: Sequence[Sequence[Dict[str, float]]], field: str = "q") -> List[float]:
    """Return the documented 24-D order (arm, gripper, arm, gripper, waist, head, base)."""
    if len(limbs) < 6:
        raise ValueError(f"expected 6 limbs, got {len(limbs)}")
    result = [joint[field] for joint in limbs[0]]
    result.append(limbs[4][0][field])
    result.extend(joint[field] for joint in limbs[1])
    result.append(limbs[4][1][field])
    result.extend(joint[field] for joint in limbs[2])
    result.extend(joint[field] for joint in limbs[3])
    result.extend(joint[field] for joint in limbs[5])
    if len(result) != 24:
        raise ValueError(f"expected 24 joints, got {len(result)}")
    return result


def nearest(items: Sequence[Dict[str, Any]], timestamps: Sequence[int], t_ns: int) -> Optional[Dict[str, Any]]:
    if not items:
        return None
    index = bisect.bisect_left(timestamps, t_ns)
    if index == 0:
        return items[0]
    if index == len(items):
        return items[-1]
    if timestamps[index] - t_ns < t_ns - timestamps[index - 1]:
        return items[index]
    return items[index - 1]


def aligned_frames(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Align every camera frame to nearest driver/end_pos and wrist cameras.

    The head camera is the master clock. Message header timestamps are used.
    """
    streams = {
        key: sorted(value, key=lambda item: item["timestamp_ns"])
        for key, value in episode.items()
        if key in ("driver", "end_pos", "head", "left", "right")
    }
    times = {key: [item["timestamp_ns"] for item in value] for key, value in streams.items()}
    rows = []
    for head in streams["head"]:
        stamp = head["timestamp_ns"]
        rows.append({
            "timestamp_ns": stamp,
            "head": head,
            "driver": nearest(streams["driver"], times["driver"], stamp),
            "end_pos": nearest(streams["end_pos"], times["end_pos"], stamp),
            "left": nearest(streams["left"], times["left"], stamp),
            "right": nearest(streams["right"], times["right"], stamp),
        })
    return rows


def export_episode(episode: Dict[str, Any], output: Path, images: bool = False) -> None:
    """Write aligned.csv and, when requested, the original JPEG bytes."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(
        json.dumps(episode["metadata"], ensure_ascii=False, indent=2), encoding="utf-8")
    rows = aligned_frames(episode)
    with (output / "aligned.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["timestamp_ns"] + [f"q_{i}" for i in range(24)] + [f"q_exp_{i}" for i in range(24)]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            driver = row["driver"]
            if not driver:
                continue
            measured = flatten_joints(driver["limbs"], "q")
            commanded = flatten_joints(driver["limbs"], "q_exp")
            writer.writerow({
                "timestamp_ns": row["timestamp_ns"],
                **{f"q_{i}": value for i, value in enumerate(measured)},
                **{f"q_exp_{i}": value for i, value in enumerate(commanded)},
            })
    if images:
        for kind in ("head", "left", "right"):
            camera_dir = output / kind
            camera_dir.mkdir(exist_ok=True)
            for index, item in enumerate(episode[kind]):
                (camera_dir / f"{index:06d}.jpg").write_bytes(item["data"])
