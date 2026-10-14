"""Assign 1012 prompts and export end-effector train/test sets.

One episode is held out of each of the twelve action groups. Prompts match the
verified collection order. Joint columns stay 24D; state_eef and action_eef are
16D.
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

ARCHIVE = Path("/iag_ad_01/ad/zhengyuan/yyzj/data/linglong_1012.tar.gz")
ROOT = Path("/iag_ad_01/ad/zhengyuan/yyzj/data/linglong_1012_processed")
TRAIN_OUT = Path("/iag_ad_01/ad/dushuyue/openpi/data/linglong_1012_eef_sft_train_15hz")
TEST_OUT = Path("/iag_ad_01/ad/dushuyue/openpi/data/linglong_1012_eef_sft_test_15hz")
SEED = 42
IMAGE_SIZE = {"head": (960, 720), "left": (960, 540), "right": (960, 540)}

def color_prompt(arm, color):
    return f"Use the {arm} arm to pick up the {color} block from the table and place it in the box."

def all_prompt(arm):
    return f"Use the {arm} arm to pick up all blocks from the table and place it in the box."

GROUPS = [
    ("right_blue", list(range(185, 200)), color_prompt("right", "blue")),
    ("right_green", list(range(200, 215)), color_prompt("right", "green")),
    ("right_orange", list(range(215, 225)), color_prompt("right", "orange")),
    ("right_yellow", list(range(225, 235)), color_prompt("right", "yellow")),
    ("right_purple", list(range(235, 245)), color_prompt("right", "purple")),
    ("left_blue", list(range(245, 255)), color_prompt("left", "blue")),
    ("left_green", list(range(255, 265)), color_prompt("left", "green")),
    ("left_orange", list(range(265, 275)), color_prompt("left", "orange")),
    ("left_yellow", [i for i in range(275, 286) if i != 277], color_prompt("left", "yellow")),
    ("left_purple", list(range(286, 296)), color_prompt("left", "purple")),
    ("left_all", list(range(296, 301)), all_prompt("left")),
    ("right_all", list(range(301, 306)), all_prompt("right")),
]

def holdout():
    rng = np.random.default_rng(SEED)
    chosen = {}
    for name, ids, _prompt in GROUPS:
        chosen[name] = int(rng.choice(np.asarray(ids)))
    return chosen

def prompt_for(episode):
    matches = [prompt for _name, ids, prompt in GROUPS if episode in ids]
    if len(matches) != 1:
        raise ValueError(f"episode {episode} belongs to {len(matches)} groups")
    return matches[0]

def extract_member(member, stream, dest):
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

def decode(raw, episode, archive_member):
    eid = str(episode)
    dest = raw.parents[1] / "decoded" / eid
    ep = parser.read_episode(raw)
    meta_episode = int(ep["metadata"].get("episode"))
    if meta_episode != episode:
        raise ValueError(f"{archive_member} metadata episode is {meta_episode}")
    for key in parser.TOPICS.values():
        if len(ep[key]) == 0:
            raise ValueError(f"{eid} {key} is empty")
    parser.export_episode(ep, dest, images=True)
    for cam, size in IMAGE_SIZE.items():
        with Image.open(dest / cam / "000000.jpg") as image:
            if image.size != size:
                raise ValueError(f"{eid} {cam} image is {image.size}, expected {size}")
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
    return {
        "episode": eid,
        "archive_member": archive_member,
        "metadata": ep["metadata"],
        "prompt": prompt_for(episode),
        "images_sha256": digest.hexdigest(),
        "counts": {key: len(ep[key]) for key in parser.TOPICS.values()},
    }

def decoded_ready(root, episode):
    dest = root / "decoded" / str(episode)
    return (dest / "aligned.csv").is_file() and (dest / "alignment.jsonl").is_file() and (dest / "head" / "000000.jpg").is_file()

def ensure_sources(root, episodes):
    root.mkdir(parents=True, exist_ok=True)
    pending = [episode for episode in episodes if not (root / "raw" / str(episode) / "bag" / "bag_0.db3").is_file()]
    if pending:
        found = set()
        with tarfile.open(ARCHIVE, "r|gz") as outer:
            for member in outer:
                if not member.isfile() or not member.name.endswith(".tar"):
                    continue
                episode = int(Path(member.name).stem.rsplit("_", 1)[-1])
                if episode not in episodes:
                    raise ValueError(f"unexpected episode {episode} in {member.name}")
                dest = root / "raw" / str(episode)
                if (dest / "bag" / "bag_0.db3").is_file():
                    src = outer.extractfile(member)
                    while src.read(1 << 20):
                        pass
                else:
                    dest.mkdir(parents=True, exist_ok=True)
                    extract_member(member, outer.extractfile(member), dest)
                found.add(episode)
                print(f"extracted {episode}", flush=True)
        missing = sorted(set(episodes) - found)
        if missing:
            raise ValueError(f"archive is missing episodes {missing}")
    records = []
    for episode in episodes:
        quality = root / "decoded" / str(episode) / "quality.json"
        if decoded_ready(root, episode) and quality.is_file():
            records.append(json.loads(quality.read_text()))
            print(f"decoded {episode} cached", flush=True)
            continue
        record = decode(root / "raw" / str(episode), episode, str(ARCHIVE))
        write(root / "decoded" / str(episode) / "quality.json", record)
        records.append(record)
        print(f"decoded {episode} prompt={record['prompt']}", flush=True)
    return records

def main():
    started = time.monotonic()
    if TRAIN_OUT.exists() or TEST_OUT.exists():
        raise SystemExit(f"refusing to overwrite {TRAIN_OUT if TRAIN_OUT.exists() else TEST_OUT}")
    episodes = [episode for _name, ids, _prompt in GROUPS for episode in ids]
    if len(episodes) != len(set(episodes)) or len(episodes) != 120:
        raise SystemExit(f"expected 120 unique episodes, got {len(episodes)}")
    split_path = ROOT / "split.json"
    if split_path.is_file():
        test_ids = {name: int(value) for name, value in json.loads(split_path.read_text())["test"].items()}
    else:
        test_ids = holdout()
        write(split_path, {"seed": SEED, "test": test_ids, "prompts": {name: prompt for name, _ids, prompt in GROUPS}})
    records = ensure_sources(ROOT, episodes)
    by_episode = {int(record["episode"]): record for record in records}
    if set(by_episode) != set(episodes):
        raise SystemExit("decoded episodes do not match the twelve groups")
    train_records, test_records = [], []
    for name, ids, _prompt in GROUPS:
        for episode in ids:
            (test_records if episode == test_ids[name] else train_records).append(by_episode[episode])
    for split, rows in (("train", train_records), ("test", test_records)):
        split_root = ROOT / split
        split_root.mkdir(parents=True, exist_ok=True)
        for name in ("raw", "decoded"):
            link = split_root / name
            if not link.exists():
                link.symlink_to(Path("..") / name, target_is_directory=True)
        write(split_root / "inventory.json", rows)
    note = f"one random episode per action group, seed {SEED}; test source ids {sorted(test_ids.values())}"
    prepare(ROOT / "train", TRAIN_OUT, prompt=GROUPS[0][2], train_fraction=1.0, split_note=note)
    prepare(ROOT / "test", TEST_OUT, prompt=GROUPS[0][2], train_fraction=0.0,
            norm_reference=TRAIN_OUT / "assets/linglong_h/norm_stats.json", split_note=note)
    write(TEST_OUT / "test_episodes.json", json.loads((TEST_OUT / "validation_episodes.json").read_text()))
    info = json.loads((TEST_OUT / "meta/info.json").read_text())
    info["splits"] = {"test": f"0:{info['total_episodes']}"}
    write(TEST_OUT / "meta/info.json", info)
    print("COMPLETE", {"test": test_ids, "seconds": round(time.monotonic() - started, 1)}, flush=True)

if __name__ == "__main__":
    main()
