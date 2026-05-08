from speakerforge.stages.stage3_vad import merge_segments, filter_segments


def test_merge_close_segments():
    segments = [(0.0, 1.0), (1.2, 2.0), (3.0, 4.0)]
    merged = merge_segments(segments, min_silence=0.3)
    assert merged == [(0.0, 2.0), (3.0, 4.0)]


def test_merge_no_merge_when_gap_large():
    segments = [(0.0, 1.0), (2.0, 3.0)]
    merged = merge_segments(segments, min_silence=0.3)
    assert merged == [(0.0, 1.0), (2.0, 3.0)]


def test_filter_segments_by_duration():
    segments = [(0.0, 0.5), (1.0, 3.0), (5.0, 14.0)]
    filtered = filter_segments(segments, min_dur=0.8, max_dur=8.0)
    assert filtered == [(1.0, 3.0)]


def test_filter_empty():
    assert filter_segments([], min_dur=0.8, max_dur=8.0) == []


def test_merge_empty():
    assert merge_segments([], min_silence=0.3) == []


def test_merge_single():
    assert merge_segments([(1.0, 2.0)], min_silence=0.3) == [(1.0, 2.0)]
