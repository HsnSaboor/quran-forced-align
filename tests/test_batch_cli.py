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


def test_validation_allows_intra_surah_split_combined_with_cuda_batch_size():
    # --intra-surah-split and --cuda-batch-size > 1 CAN be combined (see
    # pipeline.align_surahs_batched's intra_surah_split parameter and
    # engines.cuda.CUDAEngine.run_inference_batched_with_intra_surah_split)
    # -- an earlier revision of this validation rejected the combination
    # as unimplemented; this is a regression test for that no longer
    # being the case.
    args = _parse([
        "--surahs", "1", "--audio-dir", "a", "--out-dir", "b",
        "--device", "cuda", "--cuda-batch-size", "4", "--intra-surah-split",
    ])
    _validate_device_flags(args)  # must not raise


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


def test_find_audio_file_formats(tmp_path):
    from quran_forced_align.audio import find_audio_file

    audio_dir = str(tmp_path / "audio")
    import os
    os.makedirs(audio_dir, exist_ok=True)

    # 1. Zero-padded .mp3
    f1 = os.path.join(audio_dir, "001.mp3")
    open(f1, "w").close()
    assert find_audio_file(audio_dir, 1) == f1

    # 2. Zero-padded .opus
    f2 = os.path.join(audio_dir, "002.opus")
    open(f2, "w").close()
    assert find_audio_file(audio_dir, 2) == f2

    # 3. Unpadded .wav
    f3 = os.path.join(audio_dir, "3.wav")
    open(f3, "w").close()
    assert find_audio_file(audio_dir, 3) == f3

    # 4. Prefixed with reciter name (e.g. 067_alafasy.mp3)
    f67 = os.path.join(audio_dir, "067_alafasy.mp3")
    open(f67, "w").close()
    assert find_audio_file(audio_dir, 67) == f67

    # 5. Missing
    assert find_audio_file(audio_dir, 99) is None


def test_run_pipelined_batch_cpu_execution(tmp_path):
    import shutil
    import os
    from .conftest import FIXTURES_DIR, MODEL_PATH, TOKENS_PATH
    from quran_forced_align.batch_cli import run_pipelined_batch

    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model not found")

    audio_dir = str(tmp_path / "audio")
    out_dir = str(tmp_path / "out")
    os.makedirs(audio_dir, exist_ok=True)

    # Copy fixture as 001.mp3
    src_mp3 = os.path.join(FIXTURES_DIR, "001001_full.mp3")
    shutil.copy2(src_mp3, os.path.join(audio_dir, "001.mp3"))

    summary = run_pipelined_batch(
        [1],
        audio_dir=audio_dir,
        out_dir=out_dir,
        model_path=MODEL_PATH,
        tokens_path=TOKENS_PATH,
        device="cpu",
        cuda_batch_size=1,
        prefetch_workers=2,
        prefetch_batches=2,
        verbose=False,
    )

    assert summary["succeeded_count"] == 1
    assert summary["failed_count"] == 0
    assert 1 in summary["results"]
    res1 = summary["results"][1]
    assert res1["n_words"] > 0
    assert os.path.exists(res1["srt_path"])
    assert os.path.exists(res1["json_path"])

