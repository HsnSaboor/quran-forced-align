#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    int32_t stack_buf[256];
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
    
    int total_cells = T * M;
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
        const float* lp_t = log_probs + t * V;
        int8_t* bp_t = backptr + t * M;
        
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
        int8_t step = backptr[t * M + cur_s];
        cur_s -= step;
        out_path[t - 1] = cur_s;
    }
    
    if (heap_ext) free(ext_ptr);
    if (heap_alpha) free(prev_alpha < cur_alpha ? prev_alpha : cur_alpha);
    if (heap_bp) free(backptr);
    return 0;
}
