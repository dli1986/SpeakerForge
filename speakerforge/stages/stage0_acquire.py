from __future__ import annotations
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from speakerforge.bili_auth import get_or_login
from speakerforge.config import BilibiliSource, PipelineConfig, SourcesConfig


def _raw_dir(pipeline_cfg: PipelineConfig) -> Path:
    base = pipeline_cfg.stage0.get("output_root", "raw_sources")
    return Path(base) / "bilibili"


def run(sources_cfg: SourcesConfig, pipeline_cfg: PipelineConfig,
        speaker: Optional[str] = None, dry_run: bool = False) -> None:
    cookie_file = pipeline_cfg.auth.get("cookie_file", "~/.speakerforge/bili_cookies.json")
    cred = get_or_login(cookie_file)

    filtered = [s for s in sources_cfg.sources
                if speaker is None or s.speaker == speaker]
    if not filtered:
        msg = (f"No sources found for speaker '{speaker}'"
               if speaker else "sources.yaml has no entries")
        raise ValueError(msg)

    raw_dir = _raw_dir(pipeline_cfg)
    loop = asyncio.new_event_loop()
    try:
        for source in filtered:
            bv_ids = loop.run_until_complete(_resolve(source, cred))
            out_dir = raw_dir / source.speaker
            out_dir.mkdir(parents=True, exist_ok=True)
            _update_index(out_dir, source, bv_ids, status="pending")
            if dry_run:
                print(f"[dry-run] {source.speaker}: {bv_ids}")
                continue
            for bv in tqdm(bv_ids, desc=f"Downloading {source.speaker}"):
                try:
                    _download(bv, out_dir, cookie_file)
                    _set_index_status(out_dir, bv, "downloaded")
                except subprocess.CalledProcessError as exc:
                    print(f"  [WARN] Failed to download {bv}: {exc}")
                    _set_index_status(out_dir, bv, "failed")
    finally:
        loop.close()


async def _resolve(source: BilibiliSource, cred) -> list[str]:
    if source.type == "video":
        return [source.id]
    if source.type == "collection":
        return await _resolve_collection(source.uid, source.id, series_type="season", cred=cred)
    if source.type == "playlist":
        return await _resolve_collection(source.uid, source.id, series_type="series", cred=cred)
    if source.type == "user_videos":
        return await _resolve_user_videos(source.uid, source.limit or 10, cred)
    raise ValueError(f"Unknown source type: {source.type}")


async def _resolve_collection(uid: str, collection_id: str,
                               series_type: str, cred) -> list[str]:
    from bilibili_api.channel_series import ChannelSeries, ChannelSeriesType

    type_ = ChannelSeriesType.SEASON if series_type == "season" else ChannelSeriesType.SERIES
    series = ChannelSeries(uid=int(uid), type_=type_, id_=int(collection_id), credential=cred)

    bv_ids: list[str] = []
    pn = 1
    ps = 100
    while True:
        resp = await series.get_videos(pn=pn, ps=ps)
        archives = resp.get("archives", [])
        bv_ids.extend(a["bvid"] for a in archives if "bvid" in a)
        page_info = resp.get("page", {})
        total = page_info.get("total") if page_info.get("total") is not None else page_info.get("count")
        if total is not None and len(bv_ids) >= total:
            break
        if len(archives) < ps:
            break
        pn += 1

    return bv_ids


async def _resolve_user_videos(uid: str, limit: int, cred) -> list[str]:
    from bilibili_api.user import User

    user = User(uid=int(uid), credential=cred)
    bv_ids: list[str] = []
    pn = 1
    ps = min(limit, 50)
    while len(bv_ids) < limit:
        resp = await user.get_videos(pn=pn, ps=ps)
        vlist = resp.get("list", {}).get("vlist", [])
        for v in vlist:
            if "bvid" not in v:
                continue
            bv_ids.append(v["bvid"])
            if len(bv_ids) >= limit:
                break
        if len(vlist) < ps:
            break
        pn += 1

    return bv_ids[:limit]


def _download(bv_id: str, out_dir: Path, cookie_file: str) -> None:
    out_path = out_dir / f"{bv_id}.mp4"
    if out_path.exists():
        print(f"  Skip (exists): {bv_id}")
        return
    cookie_path = str(Path(cookie_file).expanduser())
    url = f"https://www.bilibili.com/video/{bv_id}/"
    cmd = [
        "yt-dlp",
        "--cookies", cookie_path,
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "--merge-output-format", "mp4",
        "--limit-rate", "1.5M",
        "--retries", "3",
        "--output", str(out_dir / f"{bv_id}.%(ext)s"),
        url,
    ]
    subprocess.run(cmd, check=True)


def _update_index(out_dir: Path, source: BilibiliSource,
                  bv_ids: list[str], status: str = "pending") -> None:
    index_path = out_dir / "source_index.json"
    index: dict = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    for bv in bv_ids:
        if bv not in index:
            index[bv] = {"speaker": source.speaker, "status": status, "duration": None}
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def _set_index_status(out_dir: Path, bv_id: str, status: str) -> None:
    index_path = out_dir / "source_index.json"
    if not index_path.exists():
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if bv_id in index:
        index[bv_id]["status"] = status
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
