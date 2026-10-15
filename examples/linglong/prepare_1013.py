"""Assign 1013 prompts and export end-effector train/test sets.

Episodes 305-354 place a rectangular block on the matching peg. Episodes
355-379 place a round block on the matching peg. The test set holds out 5
rectangular episodes and 3 round episodes. Both tasks use the right arm.
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

ARCHIVE = Path("/iag_ad_01/ad/zhengyuan/yyzj/data/linglong_10_13.tar.gz")
ROOT = Path("/iag_ad_01/ad/zhengyuan/yyzj/data/linglong_1013_processed")
TRAIN_OUT = Path("/iag_ad_01/ad/dushuyue/openpi/data/linglong_1013_eef_sft_train_15hz")
TEST_OUT = Path("/iag_ad_01/ad/dushuyue/openpi/data/linglong_1013_eef_sft_test_15hz")
SEED = 42
IMAGE_SIZE = {"head": (960, 720), "left": (960, 540), "right": (960, 540)}
RECT_IDS = list(range(305, 355))
ROUND_IDS = list(range(355, 380))
RECT_PROMPT = "Use the right arm to pick up the rectangular block from the table and place it onto the peg that matches its hole."
ROUND_PROMPT = "Use the right arm to pick up the round block from the table and place it onto the peg that matches its hole."

def holdout():
    rng = np.random.default_rng(SEED)
    rectangular = sorted(int(episode) for episode in rng.choice(np.asarray(RECT_IDS), size=5, replace=False))
    circular = sorted(int(episode) for episode in rng.choice(np.asarray(ROUND_IDS), size=3, replace=False))
    return rectangular + circular

def prompt_for(episode):
    if episode in RECT_IDS:
        return RECT_PROMPT
    if episode in ROUND_IDS:
        return ROUND_PROMPT
    raise ValueError(f"episode {episode} is outside 305-379")

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
        "shape": "rectangular" if episode in RECT_IDS else "round",
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
    episodes = RECT_IDS + ROUND_IDS
    if len(episodes) != len(set(episodes)) or len(episodes) != 75:
        raise SystemExit(f"expected 75 unique episodes, got {len(episodes)}")
    split_path = ROOT / "split.json"
    if split_path.is_file():
        test_ids = [int(episode) for episode in json.loads(split_path.read_text())["test"]]
    else:
        test_ids = holdout()
        write(split_path, {
            "seed": SEED,
            "test": test_ids,
            "rectangular": RECT_IDS,
            "round": ROUND_IDS,
            "prompts": {"rectangular": RECT_PROMPT, "round": ROUND_PROMPT},
        })
    if len(test_ids) != 8 or len(set(test_ids) & set(RECT_IDS)) != 5 or len(set(test_ids) & set(ROUND_IDS)) != 3:
        raise SystemExit(f"test split must be 5 rectangular and 3 round episodes, got {test_ids}")
    records = ensure_sources(ROOT, episodes)
    by_episode = {int(record["episode"]): record for record in records}
    if set(by_episode) != set(episodes):
        raise SystemExit("decoded episodes do not match 305-379")
    test_set = set(test_ids)
    train_records = [by_episode[episode] for episode in episodes if episode not in test_set]
    test_records = [by_episode[episode] for episode in test_ids]
    for split, rows in (("train", train_records), ("test", test_records)):
        split_root = ROOT / split
        split_root.mkdir(parents=True, exist_ok=True)
        for name in ("raw", "decoded"):
            link = split_root / name
            if not link.exists():
                link.symlink_to(Path("..") / name, target_is_directory=True)
        write(split_root / "inventory.json", rows)
    note = f"5 rectangular and 3 round episodes held out, seed {SEED}; test source ids {sorted(test_ids)}"
    prepare(ROOT / "train", TRAIN_OUT, prompt=RECT_PROMPT, train_fraction=1.0, split_note=note)
    prepare(ROOT / "test", TEST_OUT, prompt=RECT_PROMPT, train_fraction=0.0,
            norm_reference=TRAIN_OUT / "assets/linglong_h/norm_stats.json", split_note=note)
    write(TEST_OUT / "test_episodes.json", json.loads((TEST_OUT / "validation_episodes.json").read_text()))
    info = json.loads((TEST_OUT / "meta/info.json").read_text())
    info["splits"] = {"test": f"0:{info['total_episodes']}"}
    write(TEST_OUT / "meta/info.json", info)
    from dataset_catalog import update_catalog
    update_catalog(TEST_OUT)
    print("COMPLETE", {"test": sorted(test_ids), "seconds": round(time.monotonic() - started, 1)}, flush=True)

if __name__ == "__main__":
    main()
