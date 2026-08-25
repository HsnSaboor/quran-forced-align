#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))
#define NEG_INF -1e30f

double fast_token_id_levenshtein_ratio(const int32_t* a, int n, const int32_t* b, int m, double min_ratio) {
    if (n == 0 && m == 0) return 1.0;
    int max_len = n > m ? n : m;
    int min_len = n < m ? n : m;
    
    if (min_ratio >= 0.0) {
        double len_ratio = (double)min_len / (double)max_len;
        if (len_ratio < min_ratio) {
            return len_ratio;
        }
    }
    
    int row_size = m + 1;
    int32_t stack_buf[256] = {0};
    int32_t* prev_row = stack_buf;
    int32_t* cur_row = stack_buf + row_size;
    int heap_alloc = 0;
    
    if (row_size * 2 > 256) {
        prev_row = (int32_t*)malloc(row_size * 2 * sizeof(int32_t));
        cur_row = prev_row + row_size;
        heap_alloc = 1;
    }
    
    for (int j = 0; j <= m; ++j) {
        prev_row[j] = j;
    }
    
    for (int i = 1; i <= n; ++i) {
        cur_row[0] = i;
        int32_t ai = a[i - 1];
        for (int j = 1; j <= m; ++j) {
            int cost_sub = prev_row[j - 1] + (ai == b[j - 1] ? 0 : 1);
            int cost_del = prev_row[j] + 1;
            int cost_ins = cur_row[j - 1] + 1;
            cur_row[j] = MIN3(cost_sub, cost_del, cost_ins);
        }
        int32_t* tmp = prev_row;
        prev_row = cur_row;
        cur_row = tmp;
    }
    
    int distance = prev_row[m];
    if (heap_alloc) {
        free(prev_row < cur_row ? prev_row : cur_row);
    }
    
    return 1.0 - (double)distance / (double)max_len;
}

int fast_collapse_ctc_ids(const int32_t* ids, int n, int32_t blank_id, int32_t* out) {
    if (n <= 0) return 0;
    int out_count = 0;
    int32_t prev_label = blank_id;
    
    for (int i = 0; i < n; ++i) {
        int32_t cur = ids[i];
        if (cur != blank_id && cur != prev_label) {
            out[out_count++] = cur;
        }
        prev_label = cur;
    }
    return out_count;
}

// Fast pure C CTC forced alignment Viterbi DP for candidate repeat windows
int fast_ctc_forced_align(const float* log_probs, int T, int V, const int32_t* ref_ids, int L, int blank_id, int32_t* out_path) {
    if (T <= 0 || L <= 0 || T < L) return -1;
    
    int M = 2 * L + 1;
    int32_t ext[512];
    int32_t* ext_ptr = ext;
    int heap_ext = 0;
    if (M > 512) {
        ext_ptr = (int32_t*)malloc(M * sizeof(int32_t));
        heap_ext = 1;
    }
    
    for (int i = 0; i < L; ++i) {
        ext_ptr[2 * i] = blank_id;
        ext_ptr[2 * i + 1] = ref_ids[i];
    }
    ext_ptr[2 * L] = blank_id;
    
    float alpha_buf[1024];
    float* prev_alpha = alpha_buf;
    float* cur_alpha = alpha_buf + M;
    int heap_alpha = 0;
    if (2 * M > 1024) {
        prev_alpha = (float*)malloc(2 * M * sizeof(float));
        cur_alpha = prev_alpha + M;
        heap_alpha = 1;
    }
    
    size_t total_cells = (size_t)T * (size_t)M;
    
    // For large trellises (e.g. whole-surah T*M > 10M cells), use exact O(sqrt(T)*M) checkpointed DP
    if (total_cells > 10000000ULL) {
        int chunk = (int)sqrt((double)T);
        if (chunk < 2) chunk = 2;
        int num_checkpoints = (T + chunk - 1) / chunk + 2;
        float* checkpoints = (float*)malloc((size_t)num_checkpoints * (size_t)M * sizeof(float));
        
        for (int s = 0; s < M; ++s) prev_alpha[s] = NEG_INF;
        prev_alpha[0] = log_probs[ext_ptr[0]];
        if (M > 1) prev_alpha[1] = log_probs[ext_ptr[1]];
        memcpy(checkpoints, prev_alpha, M * sizeof(float));
        
        int cp_idx = 1;
        for (int t = 1; t < T; ++t) {
            const float* lp_t = log_probs + (size_t)t * (size_t)V;
            int s_min = M - 1 - 2 * (T - 1 - t);
            if (s_min < 0) s_min = 0;
            int s_max = 2 * t + 1;
            if (s_max >= M) s_max = M - 1;

            if (s_min > 0) {
                for (int s = 0; s < s_min; ++s) cur_alpha[s] = NEG_INF;
            }
            for (int s = s_min; s <= s_max; ++s) {
                float emit = lp_t[ext_ptr[s]];
                float best_val = prev_alpha[s];
                if (s > 0 && prev_alpha[s - 1] > best_val) best_val = prev_alpha[s - 1];
                if (s >= 2 && (s % 2 == 1) && ext_ptr[s] != ext_ptr[s - 2] && prev_alpha[s - 2] > best_val) best_val = prev_alpha[s - 2];
                cur_alpha[s] = (best_val > NEG_INF / 2) ? (best_val + emit) : NEG_INF;
            }
            if (s_max < M - 1) {
                for (int s = s_max + 1; s < M; ++s) cur_alpha[s] = NEG_INF;
            }
            float* tmp = prev_alpha; prev_alpha = cur_alpha; cur_alpha = tmp;
            
            if (t % chunk == 0 || t == T - 1) {
                memcpy(checkpoints + (size_t)cp_idx * (size_t)M, prev_alpha, M * sizeof(float));
                cp_idx++;
            }
        }
        
        int cur_s = M - 1;
        if (M > 1 && prev_alpha[M - 2] > prev_alpha[M - 1]) cur_s = M - 2;
        if (prev_alpha[cur_s] <= NEG_INF / 2) {
            if (heap_ext) free(ext_ptr);
            if (heap_alpha) free(prev_alpha < cur_alpha ? prev_alpha : cur_alpha);
            free(checkpoints);
            return -1;
        }
        out_path[T - 1] = cur_s;
        
        int8_t* local_bp = (int8_t*)malloc((size_t)(chunk + 2) * (size_t)M * sizeof(int8_t));
        int t_end = T - 1;
        int cp_curr = cp_idx - 1;
        
        while (t_end > 0) {
            int t_start = t_end - (t_end % chunk == 0 ? chunk : (t_end % chunk));
            if (t_start < 0) t_start = 0;
            int interval_len = t_end - t_start;
            cp_curr--;
            
            memcpy(prev_alpha, checkpoints + (size_t)cp_curr * (size_t)M, M * sizeof(float));
            for (int step = 1; step <= interval_len; ++step) {
                int t = t_start + step;
                const float* lp_t = log_probs + (size_t)t * (size_t)V;
                int8_t* bp_local = local_bp + (size_t)step * (size_t)M;
                int s_min = M - 1 - 2 * (T - 1 - t);
                if (s_min < 0) s_min = 0;
                int s_max = 2 * t + 1;
                if (s_max >= M) s_max = M - 1;

                if (s_min > 0) {
                    for (int s = 0; s < s_min; ++s) cur_alpha[s] = NEG_INF;
                }
                for (int s = s_min; s <= s_max; ++s) {
                    float emit = lp_t[ext_ptr[s]];
                    float best_val = prev_alpha[s];
                    int8_t best_step = 0;
                    if (s > 0 && prev_alpha[s - 1] > best_val) { best_val = prev_alpha[s - 1]; best_step = 1; }
                    if (s >= 2 && (s % 2 == 1) && ext_ptr[s] != ext_ptr[s - 2] && prev_alpha[s - 2] > best_val) { best_val = prev_alpha[s - 2]; best_step = 2; }
                    bp_local[s] = best_step;
                    cur_alpha[s] = (best_val > NEG_INF / 2) ? (best_val + emit) : NEG_INF;
                }
                if (s_max < M - 1) {
                    for (int s = s_max + 1; s < M; ++s) cur_alpha[s] = NEG_INF;
                }
                float* tmp = prev_alpha; prev_alpha = cur_alpha; cur_alpha = tmp;
            }
            
            for (int step = interval_len; step >= 1; --step) {
                int t = t_start + step;
                int8_t step_taken = local_bp[(size_t)step * (size_t)M + (size_t)cur_s];
                cur_s -= step_taken;
                out_path[t - 1] = cur_s;
            }
            t_end = t_start;
        }
        
        free(local_bp);
        free(checkpoints);
        if (heap_ext) free(ext_ptr);
        if (heap_alpha) free(prev_alpha < cur_alpha ? prev_alpha : cur_alpha);
        return 0;
    }
    
    int8_t backptr_stack[65536];
    int8_t* backptr = backptr_stack;
    int heap_bp = 0;
    if (total_cells > 65536) {
        backptr = (int8_t*)malloc(total_cells * sizeof(int8_t));
        heap_bp = 1;
    }
    
    for (int s = 0; s < M; ++s) {
        prev_alpha[s] = NEG_INF;
    }
    prev_alpha[0] = log_probs[ext_ptr[0]]; // blank
    if (M > 1) {
        prev_alpha[1] = log_probs[ext_ptr[1]]; // first token
    }
    
    for (int t = 1; t < T; ++t) {
        const float* lp_t = log_probs + (size_t)t * (size_t)V;
        int8_t* bp_t = backptr + (size_t)t * (size_t)M;
        
        for (int s = 0; s < M; ++s) {
            float emit = lp_t[ext_ptr[s]];
            float best_val = prev_alpha[s];
            int8_t best_step = 0; // stay
            
            if (s > 0) {
                float v1 = prev_alpha[s - 1];
                if (v1 > best_val) {
                    best_val = v1;
                    best_step = 1; // adv 1
                }
            }
            
            if (s >= 2 && (s % 2 == 1) && ext_ptr[s] != ext_ptr[s - 2]) {
                float v2 = prev_alpha[s - 2];
                if (v2 > best_val) {
                    best_val = v2;
                    best_step = 2; // adv 2
                }
            }
            
            bp_t[s] = best_step;
            cur_alpha[s] = (best_val > NEG_INF / 2) ? (best_val + emit) : NEG_INF;
        }
        
        float* tmp = prev_alpha;
        prev_alpha = cur_alpha;
        cur_alpha = tmp;
    }
    
    // Backtrace from terminal frame T-1
    int cur_s = M - 1;
    if (M > 1 && prev_alpha[M - 2] > prev_alpha[M - 1]) {
        cur_s = M - 2;
    }
    
    if (prev_alpha[cur_s] <= NEG_INF / 2) {
        if (heap_ext) free(ext_ptr);
        if (heap_alpha) free(prev_alpha < cur_alpha ? prev_alpha : cur_alpha);
        if (heap_bp) free(backptr);
        return -1; // Alignment failed
    }
    
    out_path[T - 1] = cur_s;
    for (int t = T - 1; t >= 1; --t) {
        int8_t step = backptr[(size_t)t * (size_t)M + (size_t)cur_s];
        cur_s -= step;
        out_path[t - 1] = cur_s;
    }
    
    if (heap_ext) free(ext_ptr);
    if (heap_alpha) free(prev_alpha < cur_alpha ? prev_alpha : cur_alpha);
    if (heap_bp) free(backptr);
    return 0;
}


// Fast repeat anomaly detection and phrase search engine
// Returns the total number of paths generated (total length of out_paths).
int fast_detect_and_fix_repeats_engine(
    int num_cues,
    const int32_t* cue_starts,
    const int32_t* cue_ends,
    const int32_t* cue_ayas,
    const int32_t* cue_suras,
    const int8_t* cue_is_ayah_final,
    const int32_t* cue_token_offsets,
    const int32_t* cue_token_counts,
    const int32_t* combined_token_ids,
    int combined_token_ids_len,
    const float* log_probs, // T x V
    int T,
    int V,
    const int32_t* full_greedy_ids, // T
    int blank_id,
    float confidence_floor,
    float free_decode_min_ratio_doubled,
    float free_decode_min_margin,
    int gap_artifact_max_frames,
    float gap_artifact_min_margin,
    float min_word_dur_frames,
    int max_repeat_window_words,
    int margin_val,
    double median_dur,
    double low_ratio,
    double high_ratio,
    double ayah_final_high_ratio_mult,
    // Outputs:
    int32_t* out_K, // size num_cues, initialized to 0
    int32_t* out_window_start, // size num_cues
    int32_t* out_window_end, // size num_cues
    int32_t* out_path_offsets, // size num_cues
    int32_t* out_path_lengths, // size num_cues
    int32_t* out_paths // large buffer (T*2)
) {
    int total_paths_len = 0;
    int8_t* consumed = (int8_t*)calloc(num_cues, sizeof(int8_t));
    
    // buffers for max sequence lengths
    size_t max_phrase_tokens = (size_t)combined_token_ids_len + 64;
    size_t max_ext2_len = 4 * max_phrase_tokens + 1;
    size_t buf_len = (size_t)T > max_ext2_len ? (size_t)T : max_ext2_len;
    int32_t* decoded_ids = (int32_t*)malloc(buf_len * sizeof(int32_t));
    int32_t* phrase_ids = (int32_t*)malloc(buf_len * sizeof(int32_t));
    int32_t* doubled_ids = (int32_t*)malloc(buf_len * sizeof(int32_t));
    int32_t* path2 = (int32_t*)malloc(buf_len * sizeof(int32_t));
    int32_t* ext2 = (int32_t*)malloc(buf_len * sizeof(int32_t));
    int32_t* first_seen = (int32_t*)malloc(buf_len * sizeof(int32_t));
    int32_t* last_seen = (int32_t*)malloc(buf_len * sizeof(int32_t));
    int32_t* best_path = (int32_t*)malloc(buf_len * sizeof(int32_t));

    for (int i = num_cues - 1; i >= 0; --i) {
        out_K[i] = 0;
        
        if (consumed[i]) continue;
        if (cue_token_counts[i] == 0) continue;

        int dur = cue_ends[i] - cue_starts[i] + 1;
        float high_mult = (cue_is_ayah_final[i]) ? ayah_final_high_ratio_mult : 1.0f;
        float high_cutoff = high_ratio * high_mult * (float)median_dur;
        float low_cutoff = low_ratio * (float)median_dur;
        int is_anomalous = (dur < low_cutoff || dur > high_cutoff);
        
        int gap_to_next = (i < num_cues - 1) ? (cue_starts[i + 1] - cue_ends[i] - 1) : 0;
        int has_pause_gap = (gap_to_next >= (int)median_dur);
        
        if (!is_anomalous && !has_pause_gap) continue;

        int words_left_in_aya = 1;
        while (i - words_left_in_aya >= 0 && 
               cue_ayas[i - words_left_in_aya] == cue_ayas[i] && 
               cue_suras[i - words_left_in_aya] == cue_suras[i]) {
            words_left_in_aya++;
        }
        int k_max = words_left_in_aya;
        if (k_max > 6) k_max = 6;
        if (max_repeat_window_words > 0 && k_max > max_repeat_window_words) {
            k_max = max_repeat_window_words;
        }

        int window_end = (i < num_cues - 1) ? (cue_starts[i + 1] - 1) : (T - 1);
        if (window_end > T - 1) window_end = T - 1;
        
        int j0_widest = i - k_max + 1;
        int window_start_widest = (j0_widest > 0) ? (cue_ends[j0_widest - 1] + 1) : (cue_starts[j0_widest] - margin_val);
        if (window_start_widest < 0) window_start_widest = 0;
        
        if (window_end <= window_start_widest) continue;

        int best_K = 0;
        float best_score = -1e30f;
        int best_window_start = 0;

        for (int K = 1; K <= k_max; ++K) {
            int j0 = i - K + 1;
            if (j0 < 0) break;
            int overlap = 0;
            for (int j = j0; j < i; ++j) {
                if (consumed[j]) { overlap = 1; break; }
            }
            if (overlap) break;

            int window_start = (j0 > 0) ? (cue_ends[j0 - 1] + 1) : (cue_starts[j0] - margin_val);
            if (window_start < 0) window_start = 0;
            if (window_end <= window_start) continue;
            
            int L = 0;
            for (int j = j0; j <= i; ++j) {
                int offset = cue_token_offsets[j];
                int count = cue_token_counts[j];
                for (int c = 0; c < count; ++c) {
                    phrase_ids[L++] = combined_token_ids[offset + c];
                }
            }
            for (int c = 0; c < L; ++c) {
                doubled_ids[c] = phrase_ids[c];
                doubled_ids[L + c] = phrase_ids[c];
            }
            int L_doubled = 2 * L;

            const int32_t* window_ids = full_greedy_ids + window_start;
            int window_len = window_end - window_start + 1;
            int decoded_len = fast_collapse_ctc_ids(window_ids, window_len, blank_id, decoded_ids);
            
            double ratio_doubled = fast_token_id_levenshtein_ratio(decoded_ids, decoded_len, doubled_ids, L_doubled, 0.0);
            double ratio_single = fast_token_id_levenshtein_ratio(decoded_ids, decoded_len, phrase_ids, L, 0.0);
            
            double min_margin = (K == 1) ? 0.05 : free_decode_min_margin;
            int free_decode_pass = (ratio_doubled >= free_decode_min_ratio_doubled && 
                                   (ratio_doubled - ratio_single) >= min_margin);
            if (!free_decode_pass) continue;

            int align_res = fast_ctc_forced_align(log_probs + window_start * V, window_len, V, doubled_ids, L_doubled, blank_id, path2);
            if (align_res != 0) continue;
            
            int num_states = 2 * L_doubled + 1;
            for (int s = 0; s < num_states; ++s) {
                first_seen[s] = -1;
                last_seen[s] = -1;
            }
            for (int t = 0; t < window_len; ++t) {
                int s = path2[t];
                if (s >= 0 && s < num_states) {
                    if (first_seen[s] == -1) first_seen[s] = t;
                    last_seen[s] = t;
                }
            }
            
            int timing_failed = 0;
            for (int s = 0; s < L; ++s) {
                if (first_seen[2*s+1] < 0 || last_seen[2*s+1] < 0) { timing_failed = 1; break; }
                if (first_seen[2*(L+s)+1] < 0 || last_seen[2*(L+s)+1] < 0) { timing_failed = 1; break; }
            }
            if (timing_failed) continue;
            
            int copy1_start_local = first_seen[1];
            int copy1_end_local = last_seen[2 * L - 1];
            int copy2_start_local = first_seen[2 * L + 1];
            int copy2_end_local = last_seen[4 * L - 1];
            
            int copy1_dur = copy1_end_local - copy1_start_local + 1;
            int copy2_dur = copy2_end_local - copy2_start_local + 1;
            int timing_plausible = (copy2_start_local > copy1_end_local && 
                                   copy1_dur >= min_word_dur_frames &&
                                   copy2_dur >= min_word_dur_frames);
            if (K == 1) {
                int cue_dur = cue_ends[i] - cue_starts[i] + 1;
                if (copy1_dur > 3 * cue_dur || copy2_dur > 3 * cue_dur) timing_plausible = 0;
            }
            if (!timing_plausible) continue;
            
            for (int s = 0; s < L_doubled; ++s) {
                ext2[2*s] = blank_id;
                ext2[2*s+1] = doubled_ids[s];
            }
            ext2[2*L_doubled] = blank_id;
            
            double sum1 = 0, sum2 = 0;
            for (int t = copy1_start_local; t <= copy1_end_local; ++t) {
                sum1 += log_probs[(window_start + t) * V + ext2[path2[t]]];
            }
            for (int t = copy2_start_local; t <= copy2_end_local; ++t) {
                sum2 += log_probs[(window_start + t) * V + ext2[path2[t]]];
            }
            float avg1 = (float)(sum1 / copy1_dur);
            float avg2 = (float)(sum2 / copy2_dur);
            float bilateral = (avg1 < avg2 ? avg1 : avg2);
            if (bilateral < confidence_floor) continue;
            
            // Per-word bilateral acoustic verification: ensure EVERY individual word is genuinely repeated
            int word_failed = 0;
            int tok_acc = 0;
            for (int j = j0; j <= i; ++j) {
                int count = cue_token_counts[j];
                int c1_st = first_seen[2 * tok_acc + 1];
                int c1_en = last_seen[2 * (tok_acc + count - 1) + 1];
                int c2_st = first_seen[2 * (L + tok_acc) + 1];
                int c2_en = last_seen[2 * (L + tok_acc + count - 1) + 1];
                tok_acc += count;

                int d1 = c1_en - c1_st + 1;
                int d2 = c2_en - c2_st + 1;
                if (d1 < 1 || d2 < 1) { word_failed = 1; break; }

                double w_sum1 = 0, w_sum2 = 0;
                for (int t = c1_st; t <= c1_en; ++t) {
                    w_sum1 += log_probs[(window_start + t) * V + ext2[path2[t]]];
                }
                for (int t = c2_st; t <= c2_en; ++t) {
                    w_sum2 += log_probs[(window_start + t) * V + ext2[path2[t]]];
                }
                float w_avg1 = (float)(w_sum1 / d1);
                float w_avg2 = (float)(w_sum2 / d2);
                float w_bilat = (w_avg1 < w_avg2 ? w_avg1 : w_avg2);
                if (w_bilat < confidence_floor - 0.75f) {
                    word_failed = 1;
                    break;
                }
            }
            if (word_failed) continue;
            
            int gap_frames = copy2_start_local - copy1_end_local;
            float margin_above_floor = bilateral - confidence_floor;
            if (gap_frames <= gap_artifact_max_frames && margin_above_floor < gap_artifact_min_margin) continue;
            
            float cand_score = bilateral + (float)(ratio_doubled - ratio_single) * 0.1f;
            if (cand_score > best_score) {
                best_score = cand_score;
                best_K = K;
                best_window_start = window_start;
                memcpy(best_path, path2, window_len * sizeof(int32_t));
            }
        }
        
        if (best_K > 0) {
            out_K[i] = best_K;
            out_window_start[i] = best_window_start;
            out_window_end[i] = window_end;
            out_path_offsets[i] = total_paths_len;
            
            int window_len = window_end - best_window_start + 1;
            out_path_lengths[i] = window_len;
            memcpy(out_paths + total_paths_len, best_path, window_len * sizeof(int32_t));
            total_paths_len += window_len;
            
            for (int j = i - best_K + 1; j <= i; ++j) {
                consumed[j] = 1;
            }
        }
    }
    
    free(consumed);
    free(decoded_ids);
    free(phrase_ids);
    free(doubled_ids);
    free(path2);
    free(ext2);
    free(first_seen);
    free(last_seen);
    free(best_path);
    
    return total_paths_len;
}
