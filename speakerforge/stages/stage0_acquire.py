from __future__ import annotations
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from speakerforge.bili_auth import get_or_login
from speakerforge.config import BilibiliSource, PipelineConfig, SourcesConfig

RAW_DIR = Path("raw_sources/bilibili")


def run(sources_cfg: SourcesConfig, pipeline_cfg: PipelineConfig,
        speaker: Optional[str] = None, dry_run: bool = False) -> None:
    cookie_file = pipeline_cfg.auth.get("cookie_file", "~/.speakerforge/bili_cookies.json")
    cred = get_or_login(cookie_file)

    filtered = [s for s in sources_cfg.sources
                if speaker is None or s.speaker == speaker]
    if not filtered:
        raise ValueError(f"No sources found for speaker '{speaker}'")

    loop = asyncio.new_event_loop()
    try:
        for source in filtered:
            bv_ids = loop.run_until_complete(_resolve(source, cred))
            out_dir = RAW_DIR / source.speaker
            out_dir.mkdir(parents=True, exist_ok=True)
            _update_index(out_dir, source, bv_ids)
            if dry_run:
                print(f"[dry-run] {source.speaker}: {bv_ids}")
                continue
            for bv in tqdm(bv_ids, desc=f"Downloading {source.speaker}"):
                _download(bv, out_dir, cookie_file)
    finally:
        loop.close()


async def _resolve(source: BilibiliSource, cred) -> list[str]:
    if source.type == "video":
        return [source.id]
    if source.type == "collection":
        # New-style 合集 (SEASON): uid + series id both required
        return await _resolve_collection(source.uid, source.id,
                                         series_type="season", cred=cred)
    if source.type == "playlist":
        # Old-style 列表 (SERIES): uid + series id both required
        return await _resolve_collection(source.uid, source.id,
                                         series_type="series", cred=cred)
    if source.type == "user_videos":
        return await _resolve_user_videos(source.uid, source.limit or 10, cred)
    raise ValueError(f"Unknown source type: {source.type}")


async def _resolve_collection(uid: str, collection_id: str,
                               series_type: str, cred) -> list[str]:
    """Resolve a ChannelSeries (合集 or 列表) to a list of BV IDs.

    Args:
        uid: Bilibili UP owner UID.
        collection_id: The season_id (for 'season'/合集) or series_id (for 'series'/列表).
        series_type: Either 'season' (new-style 合集) or 'series' (old-style 列表).
        cred: Bilibili Credential.

    Returns:
        List of BV ID strings.
    """
    from bilibili_api.channel_series import ChannelSeries, ChannelSeriesType

    type_ = (ChannelSeriesType.SEASON
             if series_type == "season"
             else ChannelSeriesType.SERIES)

    series = ChannelSeries(
        uid=int(uid),
        type_=type_,
        id_=int(collection_id),
        credential=cred,
    )

    bv_ids: list[str] = []
    pn = 1
    ps = 100
    while True:
        resp = await series.get_videos(pn=pn, ps=ps)
        archives = resp.get("archives", [])
        bv_ids.extend(a["bvid"] for a in archives if "bvid" in a)

        # Determine total to decide whether to paginate further
        page_info = resp.get("page", {})
        # SEASON uses page_num/page_size/total; SERIES uses pn/ps/count
        total = page_info.get("total") or page_info.get("count", 0)
        if total and len(bv_ids) >= total:
            break
        if len(archives) < ps:
            break
        pn += 1

    return bv_ids


async def _resolve_user_videos(uid: str, limit: int, cred) -> list[str]:
    """Resolve the most-recent `limit` videos uploaded by a user.

    Args:
        uid: Bilibili UP UID.
        limit: Maximum number of videos to return.
        cred: Bilibili Credential.

    Returns:
        List of BV ID strings (up to `limit` entries).
    """
    from bilibili_api.user import User

    user = User(uid=int(uid), credential=cred)
    bv_ids: list[str] = []
    pn = 1
    ps = min(limit, 50)  # API max per page; 50 is safe
    while len(bv_ids) < limit:
        resp = await user.get_videos(pn=pn, ps=ps)
        vlist = resp.get("list", {}).get("vlist", [])
        for v in vlist:
            bv_ids.append(v["bvid"])
            if len(bv_ids) >= limit:
                break
        if len(vlist) < ps:
            break  # No more pages
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


def _update_index(out_dir: Path, source: BilibiliSource, bv_ids: list[str]) -> None:
    index_path = out_dir / "source_index.json"
    index = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    for bv in bv_ids:
        if bv not in index:
            index[bv] = {"speaker": source.speaker, "status": "pending", "duration": None}
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
