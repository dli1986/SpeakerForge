import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from speakerforge.bili_auth import load_credential, save_credential, get_or_login


SAMPLE_CRED_DATA = {
    "sessdata": "sess123",
    "bili_jct": "jct456",
    "buvid3": "buv789",
    "dedeuserid": "uid000",
    "ac_time_value": "act111",
}


def test_load_credential_returns_none_if_missing(tmp_path):
    result = load_credential(str(tmp_path / "missing.json"))
    assert result is None


def test_load_credential_returns_credential(tmp_path):
    cookie_file = tmp_path / "creds.json"
    cookie_file.write_text(json.dumps(SAMPLE_CRED_DATA))
    with patch("speakerforge.bili_auth.Credential") as MockCred:
        MockCred.return_value = MagicMock()
        result = load_credential(str(cookie_file))
        MockCred.assert_called_once_with(
            sessdata="sess123",
            bili_jct="jct456",
            buvid3="buv789",
            buvid4=None,
            dedeuserid="uid000",
            ac_time_value="act111",
        )
        assert result is not None


def test_save_credential_writes_json(tmp_path):
    cookie_file = tmp_path / "creds.json"
    mock_cred = MagicMock()
    mock_cred.sessdata = "s"
    mock_cred.bili_jct = "j"
    mock_cred.buvid3 = "b"
    mock_cred.buvid4 = None
    mock_cred.dedeuserid = "d"
    mock_cred.ac_time_value = "a"
    save_credential(mock_cred, str(cookie_file))
    data = json.loads(cookie_file.read_text())
    assert data == {"sessdata": "s", "bili_jct": "j", "buvid3": "b", "buvid4": None, "dedeuserid": "d", "ac_time_value": "a"}


def test_get_or_login_returns_existing(tmp_path):
    cookie_file = tmp_path / "creds.json"
    cookie_file.write_text(json.dumps(SAMPLE_CRED_DATA))
    with patch("speakerforge.bili_auth.Credential") as MockCred:
        mock_instance = MagicMock()
        MockCred.return_value = mock_instance
        result = get_or_login(str(cookie_file), login_if_missing=False)
        assert result is mock_instance


def test_get_or_login_raises_if_missing_and_no_login(tmp_path):
    with pytest.raises(RuntimeError, match="python -m speakerforge login"):
        get_or_login(str(tmp_path / "missing.json"), login_if_missing=False)


def test_save_credential_creates_parent_dirs(tmp_path):
    cookie_file = tmp_path / "deep" / "nested" / "creds.json"
    mock_cred = MagicMock()
    mock_cred.sessdata = mock_cred.bili_jct = mock_cred.buvid3 = mock_cred.dedeuserid = mock_cred.ac_time_value = "x"
    mock_cred.buvid4 = None
    save_credential(mock_cred, str(cookie_file))
    assert cookie_file.exists()
