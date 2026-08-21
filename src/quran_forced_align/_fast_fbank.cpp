/**
 * High-performance C++ backend for deterministic Kaldi filterbank feature extraction.
 *
 * Avoids per-frame pybind11 boxing and Python interpreter loop overhead by
 * computing frames directly in native C++ using kaldi-native-fbank-core, with
 * multi-threaded OpenMP chunking across frames.
 */
#include <cstring>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "kaldi-native-fbank/csrc/online-feature.h"

extern "C" {

int compute_fbank_c(
    const float* samples,
    int num_samples,
    int tail_silence_samples,
    float* out_features
) {
    if (num_samples < 0 || tail_silence_samples < 0) {
        return 0;
    }

    const int total_samples = num_samples + tail_silence_samples;
    const int frame_length = 400; // 25ms @ 16kHz
    const int frame_shift = 160;  // 10ms @ 16kHz

    if (total_samples < frame_length) {
        return 0;
    }

    const int n_frames = (total_samples - frame_length) / frame_shift + 1;
    if (out_features == nullptr) {
        return n_frames;
    }

    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = 16000;
    opts.frame_opts.frame_shift_ms = 10;
    opts.frame_opts.frame_length_ms = 25;
    opts.frame_opts.window_type = "povey";
    opts.frame_opts.snip_edges = true;
    opts.frame_opts.dither = 0.0f; // CRITICAL: deterministic
    opts.mel_opts.num_bins = 80;
    opts.use_energy = false;

    // Parallel chunking: 2000 frames (~20s of audio) per chunk
    const int chunk_frames = 2000;
    const int num_chunks = (n_frames + chunk_frames - 1) / chunk_frames;

    #if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic, 1) if (num_chunks > 1)
    #endif
    for (int c = 0; c < num_chunks; ++c) {
        int start_frame = c * chunk_frames;
        int end_frame = (start_frame + chunk_frames < n_frames) ? (start_frame + chunk_frames) : n_frames;
        int cur_frames = end_frame - start_frame;
        if (cur_frames <= 0) continue;

        int start_sample = start_frame * frame_shift;
        int end_sample = (end_frame - 1) * frame_shift + frame_length;
        int required_samples = end_sample - start_sample;

        std::vector<float> chunk_buf(required_samples, 0.0f);
        int copy_start = start_sample;
        int copy_end = (end_sample < num_samples) ? end_sample : num_samples;
        if (copy_end > copy_start) {
            std::memcpy(chunk_buf.data(), samples + copy_start, (copy_end - copy_start) * sizeof(float));
        }

        knf::OnlineFbank fb(opts);
        fb.AcceptWaveform(16000, chunk_buf.data(), required_samples);
        fb.InputFinished();

        int ready = fb.NumFramesReady();
        int frames_to_copy = (ready < cur_frames) ? ready : cur_frames;
        for (int i = 0; i < frames_to_copy; ++i) {
            const float* frame_ptr = fb.GetFrame(i);
            std::memcpy(out_features + (start_frame + i) * 80, frame_ptr, 80 * sizeof(float));
        }
    }

    return n_frames;
}

}
