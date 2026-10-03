// Packing check: features of two sequences decoded in one batch == features decoded separately.
#include "llama.h"
#include "llama-ext.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static std::vector<float> run(llama_model * m, const std::vector<std::vector<llama_token>> & seqs, uint32_t lid, int n_embd) {
    int n = 0; for (auto & s : seqs) n += (int) s.size();
    auto cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = cp.n_ubatch = 64; cp.n_seq_max = (uint32_t) seqs.size(); cp.kv_unified = false;
    llama_context * ctx = llama_init_from_model(m, cp);
    llama_set_embeddings_layer_inp(ctx, lid, true);
    llama_batch b = llama_batch_init(n, 0, 1);
    int k = 0;
    for (size_t s = 0; s < seqs.size(); ++s) {
        for (size_t i = 0; i < seqs[s].size(); ++i, ++k) {
            b.token[k] = seqs[s][i]; b.pos[k] = (llama_pos) i; b.n_seq_id[k] = 1; b.seq_id[k][0] = (llama_seq_id) s;
            b.logits[k] = (i + 1 == seqs[s].size());
        }
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); }
    const float * h = llama_get_embeddings_layer_inp(ctx, lid);
    std::vector<float> out(h, h + (size_t) n * n_embd);
    llama_batch_free(b); llama_free(ctx);
    return out;
}

int main(int argc, char ** argv) {
    llama_backend_init();
    auto mp = llama_model_default_params();
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    const int n_embd = llama_model_n_embd(m);
    std::mt19937 g(7); std::uniform_int_distribution<> d(0, 127);
    std::vector<llama_token> A(20), B(12);
    for (auto & t : A) t = d(g);
    for (auto & t : B) t = d(g);
    auto a = run(m, {A}, 1, n_embd), bb = run(m, {B}, 1, n_embd), ab = run(m, {A, B}, 1, n_embd);
    double da = 0, db = 0;
    for (size_t i = 0; i < a.size(); ++i)  da = std::fmax(da, std::fabs(a[i]  - ab[i]));
    for (size_t i = 0; i < bb.size(); ++i) db = std::fmax(db, std::fabs(bb[i] - ab[a.size() + i]));
    printf("packed vs separate: seq A (20 tok) max|diff| %.3g, seq B (12 tok) max|diff| %.3g\n", da, db);
    llama_model_free(m); llama_backend_free();
    return 0;
}
