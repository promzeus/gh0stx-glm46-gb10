// Prototype: dump per-layer residual-stream features for all prompt tokens (DFlash training data)
// via the staging API used by draft-dflash, and cross-check against cb_eval on "l_out-<id>".
// usage: hsdump <model.gguf> <hf_layer_ids,comma> <n_tokens> [out.bin]
#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

struct cb_state {
    std::map<std::string, int> want;            // tensor name -> slot
    std::vector<std::vector<float>> got;        // slot -> [n_rows * n_embd]
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (cb_state *) ud;
    auto it = st->want.find(t->name);
    if (ask) {
        return it != st->want.end();
    }
    if (it != st->want.end() && t->type == GGML_TYPE_F32) {
        auto & dst = st->got[it->second];                 // append: one call per ubatch
        const size_t off = dst.size();
        dst.resize(off + ggml_nelements(t));
        ggml_backend_tensor_get(t, dst.data() + off, 0, ggml_nbytes(t));
    }
    return true;
}

static uint16_t f32_to_bf16(float f) {           // round-to-nearest-even
    uint32_t u; std::memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf ids n_tokens [out.bin]\n", argv[0]); return 1; }
    const char * path = argv[1];
    std::vector<int> hf_ids;
    for (char * p = strtok(argv[2], ","); p; p = strtok(nullptr, ",")) hf_ids.push_back(atoi(p));
    const int n_tok = atoi(argv[3]);
    const char * out = argc > 4 ? argv[4] : nullptr;

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mtp     = false;                     // blk.92 not needed for extraction
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) return 2;
    const int n_embd = llama_model_n_embd(model);

    cb_state st;
    for (size_t k = 0; k < hf_ids.size(); ++k) {
        st.want["l_out-" + std::to_string(hf_ids[k])] = (int) k;
    }
    st.got.resize(hf_ids.size());

    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = (uint32_t) n_tok;
    cp.n_ubatch = getenv("HSD_UB") ? (uint32_t) atoi(getenv("HSD_UB")) : (uint32_t) n_tok;
    const bool cb_only = getenv("HSD_CB_ONLY") != nullptr;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 3;

    // DFlash convention: GGUF target_layers = HF target_layer_ids + 1 (conversion/qwen.py:800);
    // input of layer (id+1) == output of HF layer id == HF hidden_states[id+1]
    if (!cb_only) for (int id : hf_ids) llama_set_embeddings_layer_inp(ctx, (uint32_t) (id + 1), true);

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::mt19937 gen(1);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    llama_batch b = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; ++i) {
        b.token[i] = dis(gen); b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = (i == n_tok - 1);          // features need no logits
    }
    b.n_tokens = n_tok;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 4; }
    llama_synchronize(ctx);

    if (cb_only) {
        for (size_t k = 0; k < hf_ids.size(); ++k) printf("cb_eval l_out-%d rows %zu (n_ubatch %u)\n", hf_ids[k], st.got[k].size() / n_embd, cp.n_ubatch);
        return 0;
    }
    std::vector<uint16_t> row((size_t) n_tok * n_embd * hf_ids.size());
    for (size_t k = 0; k < hf_ids.size(); ++k) {
        const float * h = llama_get_embeddings_layer_inp(ctx, (uint32_t) (hf_ids[k] + 1));
        const auto & ref = st.got[k];
        double max_abs = 0, max_ref = 0;
        const size_t n_ref_rows = ref.size() / n_embd;
        for (int i = 0; i < n_tok; ++i) {
            for (int j = 0; j < n_embd; ++j) {
                const float v = h[(size_t) i * n_embd + j];
                row[((size_t) i * hf_ids.size() + k) * n_embd + j] = f32_to_bf16(v);   // [tok, k*n_embd + j]
                if ((size_t) i < n_ref_rows) {
                    max_abs = std::fmax(max_abs, std::fabs(v - ref[(size_t) i * n_embd + j]));
                    max_ref = std::fmax(max_ref, std::fabs(ref[(size_t) i * n_embd + j]));
                }
            }
        }
        printf("hf_layer %d -> layer_inp %d: rows %d, cb_eval l_out rows %zu, max|diff| %.3g, max|ref| %.3g\n",
               hf_ids[k], hf_ids[k] + 1, n_tok, n_ref_rows, max_abs, max_ref);
    }
    printf("bytes/token at bf16: %zu (n_embd %d x %zu layers x 2)\n", (size_t) n_embd * hf_ids.size() * 2, n_embd, hf_ids.size());
    if (out) {
        FILE * f = fopen(out, "wb");
        fwrite(row.data(), sizeof(uint16_t), row.size(), f);
        fclose(f);
    }
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
