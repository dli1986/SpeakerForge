import asyncio
import json
from pathlib import Path
from typing import Optional

from bilibili_api.login_v2 import Credential


def load_credential(cookie_file: str) -> Optional[Credential]:
    """Load Credential from JSON file. Returns None if file does not exist."""
    path = Path(cookie_file).expanduser()
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Credential(
        sessdata=data["sessdata"],
        bili_jct=data["bili_jct"],
        buvid3=data["buvid3"],
        dedeuserid=data["dedeuserid"],
        ac_time_value=data["ac_time_value"],
    )


def save_credential(cred: Credential, cookie_file: str) -> None:
    """Persist Credential fields to JSON file."""
    path = Path(cookie_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "sessdata": cred.sessdata,
        "bili_jct": cred.bili_jct,
        "buvid3": cred.buvid3,
        "dedeuserid": cred.dedeuserid,
        "ac_time_value": cred.ac_time_value,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def get_or_login(cookie_file: str, login_if_missing: bool = True) -> Credential:
    """Return existing Credential or trigger interactive QR login."""
    cred = load_credential(cookie_file)
    if cred is not None:
        return cred
    if not login_if_missing:
        raise RuntimeError(
            f"No credential found at {cookie_file}. "
            "Run: python -m speakerforge login"
        )
    return _qr_login(cookie_file)


def _qr_login(cookie_file: str) -> Credential:
    """Interactive terminal QR-code login. Uses a persistent event loop (bilibili_api requirement)."""
    from bilibili_api.login_v2 import QrCodeLogin, QrCodeLoginChannel, QrCodeLoginState

    loop = asyncio.new_event_loop()

    async def _do_login():
        login = QrCodeLogin(platform=QrCodeLoginChannel.WEB)
        await login.start()
        print(login.get_qrcode_terminal())
        print("Scan the QR code with the Bilibili app, then confirm login.")
        while True:
            state = await login.check_state()
            if state == QrCodeLoginState.DONE:
                return login.get_credential()
            if state == QrCodeLoginState.TIMEOUT:
                raise RuntimeError("QR code timed out. Run: python -m speakerforge login")
            await asyncio.sleep(2)

    try:
        cred = loop.run_until_complete(_do_login())
    finally:
        loop.close()

    save_credential(cred, cookie_file)
    print(f"Credentials saved to {cookie_file}")
    return cred
