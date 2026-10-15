"""Rewrite the Linglong dataset page next to the datasets.

Generation and merge scripts call update_catalog() after a dataset directory
is complete. The page lists every linglong_* directory in that data folder.
"""
import html
import json
from datetime import date
from pathlib import Path

GLOSS = {
    "Use the right arm to pick up the three round parts from the table one by one and place them on the green conveyor belt.": "右臂，三个圆形零件放到绿色传送带",
    "Use the right arm to pick up the blue block from the table and place it in the box.": "右臂，蓝色方块放进盒子",
    "Use the right arm to pick up the block from the table and place it in the box.": "右臂，方块放进盒子（不写颜色）",
    "Use the right arm to pick up the green block from the table and place it in the box.": "右臂，绿色方块放进盒子",
    "Use the right arm to pick up the orange block from the table and place it in the box.": "右臂，橙色方块放进盒子",
    "Use the right arm to pick up the yellow block from the table and place it in the box.": "右臂，黄色方块放进盒子",
    "Use the right arm to pick up the purple block from the table and place it in the box.": "右臂，紫色方块放进盒子",
    "Use the left arm to pick up the blue block from the table and place it in the box.": "左臂，蓝色方块放进盒子",
    "Use the left arm to pick up the green block from the table and place it in the box.": "左臂，绿色方块放进盒子",
    "Use the left arm to pick up the orange block from the table and place it in the box.": "左臂，橙色方块放进盒子",
    "Use the left arm to pick up the yellow block from the table and place it in the box.": "左臂，黄色方块放进盒子",
    "Use the left arm to pick up the purple block from the table and place it in the box.": "左臂，紫色方块放进盒子",
    "Use the left arm to pick up all blocks from the table and place it in the box.": "左臂，桌上所有方块放进盒子",
    "Use the right arm to pick up all blocks from the table and place it in the box.": "右臂，桌上所有方块放进盒子",
    "Use the right arm to pick up the rectangular block from the table and place it onto the peg that matches its hole.": "右臂，长方形块插到对应桩",
    "Use the right arm to pick up the round block from the table and place it onto the peg that matches its hole.": "右臂，圆形块插到对应桩",
    "Use the right arm to pick up the square block from the table and place it onto the peg that matches its hole.": "右臂，方形块插到对应桩",
    "Use the right arm to pick up the triangular block from the table and place it onto the peg that matches its hole.": "右臂，三角形块插到对应桩",
}
SECTION_TITLES = {
    ("0714",): "0714 圆形零件",
    ("0923",): "0923 蓝色方块",
    ("1010",): "1010 不指定颜色",
    ("1012",): "1012 分颜色、左右手",
    ("1013",): "1013 长方与圆形插桩",
    ("1015",): "1015 方形与三角形插桩",
}


def date_key(name):
    rest = name[len("linglong_"):] if name.startswith("linglong_") else name
    dates = []
    for token in rest.split("_"):
        if len(token) == 4 and token.isdigit():
            dates.append(token)
        elif dates:
            break
    return tuple(dates)


def section_title(dates):
    if dates in SECTION_TITLES:
        return SECTION_TITLES[dates]
    return "+".join(dates) if dates else "其他"


def load_ids(path):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else None


def load_dataset(path):
    episodes_path = path / "meta/episodes.jsonl"
    if not episodes_path.exists():
        completion = {}
        if (path / "completion.json").exists():
            completion = json.loads((path / "completion.json").read_text())
        return {
            "name": path.name,
            "kind": "processed",
            "dates": date_key(path.name),
            "prompt": completion.get("prompt"),
            "train_sources": completion.get("train_sources"),
            "test_sources": completion.get("test_sources"),
            "train_segments": completion.get("train_segments"),
            "test_segments": completion.get("test_segments"),
        }
    info = json.loads((path / "meta/info.json").read_text())
    features = info.get("features", {})
    tasks = []
    tasks_path = path / "meta/tasks.jsonl"
    if tasks_path.exists():
        tasks = [json.loads(line)["task"] for line in tasks_path.read_text().splitlines() if line.strip()]
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    splits = {
        "train": load_ids(path / "train_episodes.json"),
        "val": load_ids(path / "validation_episodes.json"),
        "test": load_ids(path / "test_episodes.json"),
    }
    idsets = {key: set(value) if value is not None else None for key, value in splits.items()}
    by = {}
    for episode in episodes:
        prompt = " | ".join(episode.get("tasks") or ["(empty)"])
        if prompt not in by:
            by[prompt] = {"episodes": 0, "frames": 0, "train_ep": 0, "train_fr": 0, "val_ep": 0, "val_fr": 0, "test_ep": 0, "test_fr": 0}
        frames = int(episode.get("length") or 0)
        slot = by[prompt]
        slot["episodes"] += 1
        slot["frames"] += frames
        index = episode["episode_index"]
        for key in ("train", "val", "test"):
            if idsets[key] is not None and index in idsets[key]:
                slot[f"{key}_ep"] += 1
                slot[f"{key}_fr"] += frames
    ordered = [prompt for prompt in tasks if prompt in by]
    ordered.extend(prompt for prompt in by if prompt not in ordered)
    manifest = {}
    if (path / "manifest.json").exists():
        manifest = json.loads((path / "manifest.json").read_text())
    return {
        "name": path.name,
        "kind": "lerobot",
        "dates": date_key(path.name),
        "fps": info.get("fps"),
        "episodes": len(episodes),
        "frames": sum(int(episode.get("length") or 0) for episode in episodes),
        "eef": "state_eef" in features,
        "joint": "observation.state" in features,
        "copied_columns": bool(manifest.get("original_columns_preserved")),
        "train_n": None if splits["train"] is None else len(splits["train"]),
        "val_n": None if splits["val"] is None else len(splits["val"]),
        "test_n": None if splits["test"] is None else len(splits["test"]),
        "prompts": [(prompt, by[prompt]) for prompt in ordered],
    }


def role(dataset):
    name = dataset["name"]
    if dataset["kind"] == "processed":
        return "解码中间结果"
    if "openwam" in name:
        return "OpenWAM 导出"
    if "inlier" in name:
        return "帧筛选副本"
    if "sample" in name:
        return "训练抽样"
    if dataset.get("fps") == 30:
        return "30Hz 全集"
    if "_test_" in name:
        return "测试"
    if dataset.get("copied_columns"):
        return "末端副本"
    if len(dataset.get("dates") or []) > 1:
        return "合并"
    return "训练"


def columns(dataset):
    parts = []
    if dataset["joint"]:
        parts.append("关节 24 维")
    if dataset["eef"]:
        parts.append("末端 16 维")
    return " + ".join(parts) if parts else "未标注"


def fmt(value):
    return f"{value:,}"


def split_text(dataset):
    bits = []
    if dataset["train_n"] is not None:
        bits.append(f"训练清单 {dataset['train_n']}")
    if dataset["val_n"]:
        bits.append(f"验证清单 {dataset['val_n']}")
    if dataset["test_n"]:
        bits.append(f"测试清单 {dataset['test_n']}")
    return "，".join(bits) if bits else "无清单"


def render(datasets, written_on):
    escape = html.escape
    groups = []
    for dataset in datasets:
        key = dataset["dates"] or (dataset["name"],)
        title = section_title(dataset["dates"])
        if not groups or groups[-1][0] != key:
            groups.append((key, title, []))
        groups[-1][2].append(dataset)
    lerobot = sum(dataset["kind"] == "lerobot" for dataset in datasets)
    processed = sum(dataset["kind"] == "processed" for dataset in datasets)
    parts = [f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>玲珑数据集</title>
<style>
  :root {{ color-scheme: light; --bg:#f4f3ef; --paper:#fff; --ink:#1c1b19; --muted:#5e5a54; --line:#e2ddd4; --accent:#1f4b3a; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 "Source Han Sans SC","Noto Sans SC","PingFang SC",sans-serif; }}
  main {{ max-width:1120px; margin:0 auto; padding:32px 20px 80px; }}
  h1 {{ font-size:28px; line-height:1.25; margin:0 0 8px; }}
  h2 {{ font-size:20px; margin:36px 0 8px; }}
  h3 {{ font-size:16px; margin:0 0 4px; }}
  p {{ color:var(--muted); margin:0 0 12px; }}
  a {{ color:var(--accent); text-decoration:none; border-bottom:1px solid var(--line); }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:0.92em; }}
  .stats {{ display:flex; gap:16px; flex-wrap:wrap; margin:20px 0 8px; }}
  .stat {{ background:var(--paper); border:1px solid var(--line); padding:12px 16px; min-width:140px; }}
  .stat b {{ display:block; font-size:22px; color:var(--ink); }}
  .stat span {{ color:var(--muted); font-size:13px; }}
  nav.toc {{ display:flex; flex-wrap:wrap; gap:8px 14px; margin:8px 0 20px; }}
  input[type=search] {{ width:min(420px,100%); padding:8px 10px; border:1px solid var(--line); background:var(--paper); font:inherit; }}
  table {{ width:100%; border-collapse:collapse; background:var(--paper); margin:8px 0 4px; }}
  th, td {{ text-align:left; vertical-align:top; padding:8px 10px; border-bottom:1px solid var(--line); }}
  th {{ font-size:13px; color:var(--muted); font-weight:600; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  .prompt {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:13px; line-height:1.45; }}
  .tag {{ display:inline-block; font-size:12px; color:var(--accent); border:1px solid var(--line); padding:0 6px; margin-right:6px; }}
  section.card {{ background:var(--paper); border:1px solid var(--line); padding:14px 16px 8px; margin:12px 0; }}
  .hidden {{ display:none; }}
</style>
</head>
<body>
<main>
<h1>玲珑数据集</h1>
<p>按目录列出每个 <code>linglong_*</code> 数据集。每种 prompt 的条数和帧数来自该目录的 <code>meta/episodes.jsonl</code>，训练、验证、测试清单单独计。生成或合并数据集结束时会重写本页。快照时间 {escape(written_on)}。合并目录和副本里的 episode 与来源目录重复，不同目录的数字不要相加。</p>
<div class="stats">
  <div class="stat"><b>{lerobot}</b><span>个 LeRobot 目录</span></div>
  <div class="stat"><b>{processed}</b><span>个解码中间目录</span></div>
</div>
<label for="q">筛选目录名或 prompt</label><br>
<input id="q" type="search" placeholder="例如 1012、blue、插桩" autocomplete="off">
<nav class="toc">
''']
    for key, title, members in groups:
        anchor = "g-" + "-".join(key)
        parts.append(f'<a href="#{anchor}">{escape(title)}（{len(members)}）</a>')
    parts.append("</nav>")
    for key, title, members in groups:
        anchor = "g-" + "-".join(key)
        parts.append(f'<h2 id="{anchor}">{escape(title)}</h2>')
        for dataset in members:
            blob = dataset["name"] + " " + (dataset.get("prompt") or "")
            if dataset["kind"] == "lerobot":
                blob += " " + " ".join(prompt for prompt, _stat in dataset["prompts"])
                blob += " " + " ".join(GLOSS.get(prompt, "") for prompt, _stat in dataset["prompts"])
            parts.append(f'<section class="card" id="{escape(dataset["name"])}" data-text="{escape(blob)}">')
            parts.append(f'<h3><code>{escape(dataset["name"])}</code></h3>')
            if dataset.get("error"):
                parts.append(f'<p>读取这个目录失败：{escape(dataset["error"])}</p>')
            elif dataset["kind"] == "processed":
                bits = [f'<span class="tag">{escape(role(dataset))}</span>']
                if dataset.get("train_sources") is not None:
                    bits.append(f'训练源 {dataset["train_sources"]}')
                if dataset.get("train_segments") is not None:
                    bits.append(f'训练段 {dataset["train_segments"]}')
                if dataset.get("test_sources") is not None:
                    bits.append(f'测试源 {dataset["test_sources"]}')
                if dataset.get("test_segments") is not None:
                    bits.append(f'测试段 {dataset["test_segments"]}')
                parts.append("<p>" + " · ".join(bits) + "</p>")
                if dataset.get("prompt"):
                    gloss = GLOSS.get(dataset["prompt"], "")
                    if gloss:
                        parts.append(f"<p>{escape(gloss)}</p>")
                    parts.append(f'<p class="prompt">{escape(dataset["prompt"])}</p>')
                parts.append("<p>解码后的中间目录，没有 LeRobot episode 表，所以没有帧数。</p>")
            else:
                parts.append(
                    f'<p><span class="tag">{escape(role(dataset))}</span>'
                    f'{dataset["fps"]} Hz · {escape(columns(dataset))} · {fmt(dataset["episodes"])} 条 / {fmt(dataset["frames"])} 帧'
                    f' · {escape(split_text(dataset))} · {len(dataset["prompts"])} 种 prompt</p>'
                )
                show_held = any(stat["val_ep"] or stat["test_ep"] for _prompt, stat in dataset["prompts"])
                headers = "<th>prompt</th><th class=\"num\">条数</th><th class=\"num\">帧数</th><th class=\"num\">训练清单条数</th><th class=\"num\">训练清单帧数</th>"
                if show_held:
                    headers += "<th class=\"num\">验证或测试</th>"
                parts.append(f"<table><thead><tr>{headers}</tr></thead><tbody>")
                for prompt, stat in dataset["prompts"]:
                    held = []
                    if stat["val_ep"]:
                        held.append(f"验证 {stat['val_ep']} 条 / {fmt(stat['val_fr'])} 帧")
                    if stat["test_ep"]:
                        held.append(f"测试 {stat['test_ep']} 条 / {fmt(stat['test_fr'])} 帧")
                    gloss = GLOSS.get(prompt, "")
                    label = f"<div>{escape(gloss)}</div>" if gloss else ""
                    row = (
                        "<tr>"
                        f"<td>{label}<div class=\"prompt\">{escape(prompt)}</div></td>"
                        f"<td class=\"num\">{fmt(stat['episodes'])}</td>"
                        f"<td class=\"num\">{fmt(stat['frames'])}</td>"
                        f"<td class=\"num\">{fmt(stat['train_ep']) if dataset['train_n'] is not None else '—'}</td>"
                        f"<td class=\"num\">{fmt(stat['train_fr']) if dataset['train_n'] is not None else '—'}</td>"
                    )
                    if show_held:
                        row += f"<td class=\"num\">{'；'.join(held) if held else '—'}</td>"
                    parts.append(row + "</tr>")
                parts.append("</tbody></table>")
            parts.append("</section>")
    parts.append('''<script>
const q = document.getElementById('q');
q.addEventListener('input', () => {
  const needle = q.value.trim().toLowerCase();
  document.querySelectorAll('section.card').forEach(section => {
    const text = (section.dataset.text || '').toLowerCase();
    section.classList.toggle('hidden', needle && !text.includes(needle));
  });
});
</script>
</main>
</body>
</html>
''')
    return "".join(parts)


def update_catalog(path):
    """Rewrite linglong_datasets.html beside the linglong_* directories."""
    path = Path(path).resolve()
    root = path.parent if path.name.startswith("linglong_") else path
    datasets = []
    for directory in sorted(root.glob("linglong_*")):
        if not directory.is_dir():
            continue
        try:
            datasets.append(load_dataset(directory))
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            datasets.append({
                "name": directory.name,
                "kind": "processed",
                "dates": date_key(directory.name),
                "prompt": None,
                "train_sources": None,
                "test_sources": None,
                "train_segments": None,
                "test_segments": None,
                "error": str(exc),
            })
    datasets.sort(key=lambda dataset: (max(dataset["dates"]) if dataset["dates"] else dataset["name"], len(dataset["dates"]), dataset["dates"], dataset["name"]))
    destination = root / "linglong_datasets.html"
    destination.write_text(render(datasets, date.today().isoformat()), encoding="utf-8")
    print(f"catalog {destination}", flush=True)
    return destination
