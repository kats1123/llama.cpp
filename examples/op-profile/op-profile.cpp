// LOCAL (2026-11-21): per-operation timing for one prefill ubatch and a few decode steps at a given context depth.
// Uses the scheduler's eval callback: every node is computed and synchronized on its own, and the wall time between
// callbacks is charged to that node. Adds ~10-30 us per node, so glue ops look a bit heavier than they are, but the
// big operations (attention, expert matmuls) rank correctly.
//
//   llama-op-profile -m model.gguf -c 70000 -ub 512 -b 512 -ngl 99 -fa on -ctk q8_0 -ctv q8_0   (env PROF_DEPTH=65536)
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

struct prof_data {
    bool active = false;
    bool seen_first = false;              // first "ask" of a decode call = graph built, allocated and inputs set
    double host_us = 0.0;                 // time from llama_process() to the first node
    std::chrono::steady_clock::time_point call_start;
    std::chrono::steady_clock::time_point last;
    std::map<std::string, std::pair<double, int>> by_op;
    std::map<std::string, std::pair<double, int>> by_name;
    double total = 0.0;
    int nodes = 0;
};

static std::string name_key(const char * n) {
    std::string s(n);
    const size_t dash = s.find('-');      // "attn_out-12" -> "attn_out"
    if (dash != std::string::npos) {
        s = s.substr(0, dash);
    }
    const size_t sp = s.find(' ');        // "x (view)" -> "x"
    if (sp != std::string::npos) {
        s = s.substr(0, sp);
    }
    return s.empty() ? "(unnamed)" : s;
}

static bool prof_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * d = (prof_data *) ud;
    if (ask) {
        if (d->active && !d->seen_first) {
            d->seen_first = true;
            const auto now = std::chrono::steady_clock::now();
            d->host_us += std::chrono::duration<double, std::micro>(now - d->call_start).count();
            d->last = now;
        }
        return d->active;
    }
    const auto now = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(now - d->last).count();
    d->last = now;
    const std::string op = ggml_op_desc(t);
    if (getenv("PROF_BIG") && ggml_nbytes(t) > (size_t) 50*1024*1024) {
        static std::map<std::string, int> seen;
        char key[256];
        snprintf(key, sizeof(key), "%s %s %s [%lld,%lld,%lld,%lld] %.1f MB", op.c_str(), name_key(t->name).c_str(), ggml_type_name(t->type),
                 (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], ggml_nbytes(t)/1048576.0);
        if (seen[key]++ == 0) {
            fprintf(stderr, "BIG %s\n", key);
        }
    }
    auto & a = d->by_op[op];
    a.first += us; a.second++;
    auto & b = d->by_name[op + "  " + name_key(t->name)];
    b.first += us; b.second++;
    d->total += us;
    d->nodes++;
    return true;
}

static void report(const prof_data & d, const char * title, int n_tokens) {
    printf("\n=== %s: %d nodes, %.1f ms in nodes + %.1f ms host before the first node (graph build, alloc, inputs)\n",
           title, d.nodes, d.total/1000.0, d.host_us/1000.0);
    printf("    (%.0f tok/s incl. profiling overhead)\n", n_tokens/((d.total + d.host_us)/1e6));
    auto dump = [&](const std::map<std::string, std::pair<double, int>> & m, const char * what, size_t top) {
        std::vector<std::pair<std::string, std::pair<double, int>>> v(m.begin(), m.end());
        std::sort(v.begin(), v.end(), [](const auto & x, const auto & y) { return x.second.first > y.second.first; });
        printf("--- by %s (top %zu)\n", what, top);
        for (size_t i = 0; i < std::min(top, v.size()); ++i) {
            printf("  %6.1f%%  %9.2f ms  %5d x  %s\n", 100.0*v[i].second.first/d.total, v[i].second.first/1000.0,
                   v[i].second.second, v[i].first.c_str());
        }
    };
    dump(d.by_op, "op", 14);
    dump(d.by_name, "op + tensor name", 28);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    prof_data pd;
    params.cb_eval = prof_cb;
    params.cb_eval_user_data = &pd;
    params.warmup = false;

    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    const llama_model * model = init->model();
    if (!ctx || !model) {
        fprintf(stderr, "init failed\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int depth   = getenv("PROF_DEPTH") ? atoi(getenv("PROF_DEPTH")) : 32768;
    const int n_ub    = (int) llama_n_ubatch(ctx);
    const int n_b     = (int) llama_n_batch(ctx);

    std::mt19937 rng(1234);
    auto chunk = [&](int n) {
        llama_tokens toks(n);
        for (auto & t : toks) {
            t = (llama_token) (1000 + rng() % (uint32_t) (n_vocab - 2000));
        }
        common_batch batch = common_batch_get_one(ctx, toks);
        pd.seen_first = false;
        pd.call_start = std::chrono::steady_clock::now();
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "decode failed\n");
            exit(1);
        }
    };

    const auto t0 = std::chrono::steady_clock::now();
    for (int pos = 0; pos < depth; ) {
        const int n = std::min(n_b, depth - pos);
        chunk(n);
        pos += n;
    }
    printf("filled %d tokens in %.1f s (unprofiled)\n", depth,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    pd.active = true; pd.last = std::chrono::steady_clock::now();
    chunk(n_ub);
    pd.active = false;
    char title[128];
    snprintf(title, sizeof(title), "PREFILL one ubatch of %d tokens at depth %d", n_ub, depth);
    report(pd, title, n_ub);

    pd = prof_data();
    const int n_dec = 8;
    pd.active = true; pd.last = std::chrono::steady_clock::now();
    for (int i = 0; i < n_dec; ++i) {
        chunk(1);
    }
    pd.active = false;
    snprintf(title, sizeof(title), "DECODE %d single tokens at depth %d", n_dec, depth + n_ub);
    report(pd, title, n_dec);

    llama_backend_free();
    return 0;
}
