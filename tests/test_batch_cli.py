"""Unit tests for batch_cli.py's pure-Python argument parsing/validation
logic (parse_surah_list, _chunked, and main()'s --cuda-batch-size/
--intra-surah-split validation branches) -- none of this needs a GPU, a
model file, or ProcessPoolExecutor to actually run an alignment, but this
was found to have ZERO test coverage in code review despite being the
validation layer for the --cuda-batch-size/--intra-surah-split features.
"""
import pytest

from quran_forced_align.batch_cli import _chunked, _validate_device_flags, build_parser, parse_surah_list


def test_parse_surah_list_range():
    assert parse_surah_list("67-71") == [67, 68, 69, 70, 71]


def test_parse_surah_list_single_surah_range():
    assert parse_surah_list("5-5") == [5]


def test_parse_surah_list_comma_list():
    assert parse_surah_list("67,68,69") == [67, 68, 69]


def test_parse_surah_list_comma_list_with_whitespace():
    assert parse_surah_list(" 67, 68 ,69 ") == [67, 68, 69]


def test_parse_surah_list_single_value_no_separator():
    assert parse_surah_list("42") == [42]


def test_parse_surah_list_invalid_range_end_before_start():
    with pytest.raises(ValueError, match="end < start"):
        parse_surah_list("71-67")


def test_chunked_exact_multiple():
    assert _chunked([1, 2, 3, 4, 5, 6], 3) == [[1, 2, 3], [4, 5, 6]]


def test_chunked_with_remainder():
    assert _chunked([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4) == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]]


def test_chunked_size_larger_than_input():
    assert _chunked([1, 2, 3], 10) == [[1, 2, 3]]


def test_chunked_size_one():
    assert _chunked([1, 2, 3], 1) == [[1], [2], [3]]


def test_chunked_empty_input():
    assert _chunked([], 4) == []


def _parse(argv):
    return build_parser().parse_args(argv)


def test_default_device_is_cpu_and_batch_size_is_one():
    args = _parse(["--surahs", "1-3", "--audio-dir", "a", "--out-dir", "b"])
    assert args.device == "cpu"
    assert args.cuda_batch_size == 1
    assert args.intra_surah_split is False


def test_cuda_batch_size_and_intra_surah_split_flags_parse():
    args = _parse([
        "--surahs", "1-3", "--audio-dir", "a", "--out-dir", "b",
        "--device", "cuda", "--cuda-batch-size", "8", "--intra-surah-split",
    ])
    assert args.device == "cuda"
    assert args.cuda_batch_size == 8
    assert args.intra_surah_split is True


def test_validation_rejects_cuda_batch_size_without_cuda_device():
    args = _parse(["--surahs", "1", "--audio-dir", "a", "--out-dir", "b", "--cuda-batch-size", "4"])
    with pytest.raises(SystemExit, match="requires --device cuda"):
        _validate_device_flags(args)


def test_validation_rejects_cuda_batch_size_below_one():
    args = _parse([
        "--surahs", "1", "--audio-dir", "a", "--out-dir", "b",
        "--device", "cuda", "--cuda-batch-size", "0",
    ])
    with pytest.raises(SystemExit, match="must be >= 1"):
        _validate_device_flags(args)


def test_validation_rejects_intra_surah_split_without_cuda_device():
    args = _parse(["--surahs", "1", "--audio-dir", "a", "--out-dir", "b", "--intra-surah-split"])
    with pytest.raises(SystemExit, match="requires --device cuda"):
        _validate_device_flags(args)


def test_validation_rejects_intra_surah_split_combined_with_cuda_batch_size():
    args = _parse([
        "--surahs", "1", "--audio-dir", "a", "--out-dir", "b",
        "--device", "cuda", "--cuda-batch-size", "4", "--intra-surah-split",
    ])
    with pytest.raises(SystemExit, match="cannot be combined"):
        _validate_device_flags(args)


def test_validation_passes_for_valid_cuda_batch_size_alone():
    args = _parse([
        "--surahs", "1", "--audio-dir", "a", "--out-dir", "b",
        "--device", "cuda", "--cuda-batch-size", "4",
    ])
    _validate_device_flags(args)  # must not raise


def test_validation_passes_for_valid_intra_surah_split_alone():
    args = _parse([
        "--surahs", "1", "--audio-dir", "a", "--out-dir", "b",
        "--device", "cuda", "--intra-surah-split",
    ])
    _validate_device_flags(args)  # must not raise


def test_validation_passes_for_plain_cpu_defaults():
    args = _parse(["--surahs", "1", "--audio-dir", "a", "--out-dir", "b"])
    _validate_device_flags(args)  # must not raise
