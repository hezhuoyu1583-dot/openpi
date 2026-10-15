"""Assign 1015 prompts and export end-effector train/test sets.

Collection order is chronological. The first 25 episodes place a square block
on the matching peg, and the last 25 place a triangular block. Wording matches
the 1013 prompts except for that shape word. The test set holds out 3 episodes
of each shape. Both tasks use the right arm.
"""
import hashlib
import io
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import linglong_parser as parser
from prepare_sft import prepare, write

ARCHIVE = Path("/iag_ad_01/ad/zhengyuan/yyzj/data/linglong_1015.tar.gz")
ROOT = Path("/iag_ad_01/ad/zhengyuan/yyzj/data/linglong_1015_processed")
TRAIN_OUT = Path("/iag_ad_01/ad/dushuyue/openpi/data/linglong_1015_eef_sft_train_15hz")
TEST_OUT = Path("/iag_ad_01/ad/dushuyue/openpi/data/linglong_1015_eef_sft_test_15hz")
SEED = 42
IMAGE_SIZE = {"head": (960, 720), "left": (960, 540), "right": (960, 540)}
SQUARE_PROMPT = "Use the right arm to pick up the square block from the table and place it onto the peg that matches its hole."
TRIANGLE_PROMPT = "Use the right arm to pick up the triangular block from the table and place it onto the peg that matches its hole."


def episode_id(member_name):
    parent = Path(member_name).parent.name
    if not parent or parent in {".", ".."}:
        raise ValueError(f"episode archive has no timestamp directory: {member_name}")
    return parent


def holdout(square_ids, triangle_ids):
    rng = np.random.default_rng(SEED)
    square = sorted(str(episode) for episode in rng.choice(np.asarray(square_ids), size=3, replace=False))
    triangle = sorted(str(episode) for episode in rng.choice(np.asarray(triangle_ids), size=3, replace=False))
    return square + triangle


def prompt_for(episode, square_ids):
    if episode in square_ids:
        return SQUARE_PROMPT
    return TRIANGLE_PROMPT


def extract_member(stream, dest):
    inner = tarfile.open(fileobj=stream, mode="r|")
    for item in inner:
        if not item.isfile():
            continue
        rel = Path(item.name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe archive path {item.name}")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with inner.extractfile(item) as src, target.open("wb") as out:
            shutil.copyfileobj(src, out)


def discover(root):
    root.mkdir(parents=True, exist_ok=True)
    found = {}
    with tarfile.open(ARCHIVE, "r|gz") as outer:
        for member in outer:
            if not member.isfile() or not member.name.endswith(".tar"):
                continue
            episode = episode_id(member.name)
            if episode in found:
                raise ValueError(f"duplicate timestamp directory {episode}")
            found[episode] = member.name
            dest = root / "raw" / episode
            if (dest / "bag" / "bag_0.db3").is_file():
                src = outer.extractfile(member)
                while src.read(1 << 20):
                    pass
            else:
                dest.mkdir(parents=True, exist_ok=True)
                extract_member(outer.extractfile(member), dest)
            print(f"extracted {episode} {Path(member.name).name}", flush=True)
    ordered = sorted(found)
    if len(ordered) != 50:
        raise ValueError(f"expected 50 episodes, found {len(ordered)}")
    if not Path(found[ordered[24]]).name.endswith("1014_16.tar"):
        raise ValueError(f"episode 25 is {found[ordered[24]]}, expected 积木插拔1014_16")
    if not Path(found[ordered[25]]).name.endswith("1014_17.tar"):
        raise ValueError(f"episode 26 is {found[ordered[25]]}, expected 积木插拔1014_17")
    return ordered, found


def decode(raw, episode, archive_member, square_ids):
    dest = raw.parents[1] / "decoded" / episode
    ep = parser.read_episode(raw)
    meta_episode = str(ep["metadata"].get("episode"))
    suffix = Path(archive_member).stem.rsplit("_", 1)[-1]
    if meta_episode != suffix:
        raise ValueError(f"{archive_member} metadata episode is {meta_episode}")
    for key in parser.TOPICS.values():
        if len(ep[key]) == 0:
            raise ValueError(f"{episode} {key} is empty")
    parser.export_episode(ep, dest, images=True)
    for cam, size in IMAGE_SIZE.items():
        with Image.open(dest / cam / "000000.jpg") as image:
            if image.size != size:
                raise ValueError(f"{episode} {cam} image is {image.size}, expected {size}")
    indices = {key: {id(item): i for i, item in enumerate(ep[key])} for key in ("head", "left", "right")}
    with (dest / "alignment.jsonl").open("w") as handle:
        for row in parser.aligned_frames(ep):
            handle.write(json.dumps({
                "timestamp_ns": row["timestamp_ns"],
                "source_timestamps": {key: row[key]["timestamp_ns"] for key in parser.TOPICS.values()},
                "images": {key: f"{key}/{indices[key][id(row[key])]:06d}.jpg" for key in indices},
            }) + "\n")
    digest = hashlib.sha256()
    for cam in indices:
        for item in ep[cam]:
            with Image.open(io.BytesIO(item["data"])) as image:
                image.load()
            digest.update(item["data"])
    shape = "square" if episode in square_ids else "triangular"
    return {
        "episode": episode,
        "archive_member": archive_member,
        "metadata": ep["metadata"],
        "prompt": SQUARE_PROMPT if shape == "square" else TRIANGLE_PROMPT,
        "shape": shape,
        "images_sha256": digest.hexdigest(),
        "counts": {key: len(ep[key]) for key in parser.TOPICS.values()},
    }


def decoded_ready(root, episode):
    dest = root / "decoded" / episode
    return (dest / "aligned.csv").is_file() and (dest / "alignment.jsonl").is_file() and (dest / "head" / "000000.jpg").is_file()


def ensure_sources(root, ordered, members, square_ids):
    square_set = set(square_ids)
    records = []
    for episode in ordered:
        quality = root / "decoded" / episode / "quality.json"
        if decoded_ready(root, episode) and quality.is_file():
            records.append(json.loads(quality.read_text()))
            print(f"decoded {episode} cached", flush=True)
            continue
        record = decode(root / "raw" / episode, episode, members[episode], square_set)
        write(quality, record)
        records.append(record)
        print(f"decoded {episode} shape={record['shape']}", flush=True)
    return records


def main():
    started = time.monotonic()
    if TRAIN_OUT.exists() or TEST_OUT.exists():
        raise SystemExit(f"refusing to overwrite {TRAIN_OUT if TRAIN_OUT.exists() else TEST_OUT}")
    ordered, members = discover(ROOT)
    square_ids = ordered[:25]
    triangle_ids = ordered[25:]
    split_path = ROOT / "split.json"
    if split_path.is_file():
        test_ids = [str(episode) for episode in json.loads(split_path.read_text())["test"]]
    else:
        test_ids = holdout(square_ids, triangle_ids)
        write(split_path, {
            "seed": SEED,
            "test": test_ids,
            "square": square_ids,
            "triangular": triangle_ids,
            "members": members,
            "prompts": {"square": SQUARE_PROMPT, "triangular": TRIANGLE_PROMPT},
        })
    if len(test_ids) != 6 or len(set(test_ids) & set(square_ids)) != 3 or len(set(test_ids) & set(triangle_ids)) != 3:
        raise SystemExit(f"test split must be 3 square and 3 triangular episodes, got {test_ids}")
    records = ensure_sources(ROOT, ordered, members, square_ids)
    by_episode = {record["episode"]: record for record in records}
    if set(by_episode) != set(ordered):
        raise SystemExit("decoded episodes do not match the archive")
    for episode, record in by_episode.items():
        expected = prompt_for(episode, set(square_ids))
        if record["prompt"] != expected:
            raise SystemExit(f"{episode} prompt is {record['prompt']}")
    test_set = set(test_ids)
    train_records = [by_episode[episode] for episode in ordered if episode not in test_set]
    test_records = [by_episode[episode] for episode in test_ids]
    for split, rows in (("train", train_records), ("test", test_records)):
        split_root = ROOT / split
        split_root.mkdir(parents=True, exist_ok=True)
        for name in ("raw", "decoded"):
            link = split_root / name
            if not link.exists():
                link.symlink_to(Path("..") / name, target_is_directory=True)
        write(split_root / "inventory.json", rows)
    note = f"3 square and 3 triangular episodes held out, seed {SEED}; test source ids {test_ids}"
    prepare(ROOT / "train", TRAIN_OUT, prompt=SQUARE_PROMPT, train_fraction=1.0, split_note=note)
    prepare(ROOT / "test", TEST_OUT, prompt=SQUARE_PROMPT, train_fraction=0.0,
            norm_reference=TRAIN_OUT / "assets/linglong_h/norm_stats.json", split_note=note)
    write(TEST_OUT / "test_episodes.json", json.loads((TEST_OUT / "validation_episodes.json").read_text()))
    info = json.loads((TEST_OUT / "meta/info.json").read_text())
    info["splits"] = {"test": f"0:{info['total_episodes']}"}
    write(TEST_OUT / "meta/info.json", info)
    from dataset_catalog import update_catalog
    update_catalog(TEST_OUT)
    print("COMPLETE", {"test": test_ids, "seconds": round(time.monotonic() - started, 1)}, flush=True)


if __name__ == "__main__":
    main()
