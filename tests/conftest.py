import numpy as np
import pytest
import soundfile as sf
from pathlib import Path


@pytest.fixture
def make_wav(tmp_path):
    """Factory: create a synthetic WAV file."""
    def _make(filename: str, duration: float = 3.0, sr: int = 24000) -> Path:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = np.random.randn(int(duration * sr)).astype(np.float32) * 0.1
        sf.write(str(path), samples, sr)
        return path
    return _make
