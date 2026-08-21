#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

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
