#!/usr/bin/env python3
"""
offline_dflash_patch.py — добавляет OFFLINE-backend в SpecForge DFlash, чтобы train_dflash
учился на ПРЕД-СЧИТАННЫХ hidden-states (снятых нашим vLLM-экстрактором на GB10 с NVFP4 v4),
БЕЗ живого teacher (hf деквантует→OOM, sglang нет на sm_121a).

Механика: OfflineDFlashTargetModel реализует тот же интерфейс, что HF/SGLang backend:
  generate_dflash_data(input_ids, attention_mask, loss_mask) -> DFlashTargetOutput
Ищет пред-считанные hidden_states по хэшу непадженных input_ids (устойчиво к шаффлу/паддингу),
ленивая загрузка с диска, ре-паддинг под seq текущего батча.

Запуск ВНУТРИ контейнера: python3 offline_dflash_patch.py  (правит /tmp/SpecForge на месте)
Затем: train_dflash.py --target-model-backend offline --offline-hidden-dir <dir> ...
"""
import re, os

ROOT = os.environ.get("SPECFORGE_ROOT", "/tmp/SpecForge")
TM = os.path.join(ROOT, "specforge/modeling/target/dflash_target_model.py")
TR = os.path.join(ROOT, "scripts/train_dflash.py")

OFFLINE_CLASS = '''

# ===== OFFLINE backend (GB10: hidden-states пред-считаны vLLM-экстрактором) =====
import os
import hashlib as _hashlib
class OfflineDFlashTargetModel(DFlashTargetModel):
    """Читает пред-считанные hidden_states (конкат capture-слоёв) с диска по хэшу input_ids."""
    def __init__(self, hidden_dir: str):
        super().__init__()
        self.hidden_dir = hidden_dir

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, torch_dtype=None,
                        device: str = None, cache_dir=None, hidden_dir: str = None, **kwargs):
        hd = hidden_dir or os.environ.get("OFFLINE_HIDDEN_DIR")
        assert hd and os.path.isdir(hd), f"OFFLINE_HIDDEN_DIR не задан/не найден: {hd}"
        return cls(hd)

    def set_capture_layers(self, layer_ids):
        super().set_capture_layers(layer_ids)  # no-op: экстрактор уже снял эти слои

    @staticmethod
    def _key(ids_1d):
        return _hashlib.md5(",".join(map(str, ids_1d)).encode()).hexdigest()

    def generate_dflash_data(self, input_ids, attention_mask, loss_mask) -> DFlashTargetOutput:
        import torch
        B, S = input_ids.shape
        rows = []
        for b in range(B):
            am = attention_mask[b].bool()
            real = input_ids[b][am].tolist()
            f = os.path.join(self.hidden_dir, self._key(real) + ".pt")
            d = torch.load(f, map_location="cpu")
            hs = d["hidden_states"]                      # [real_len, H*ncap]
            pad = torch.zeros(S, hs.shape[-1], dtype=hs.dtype)
            idx = am.nonzero(as_tuple=True)[0]
            pad[idx] = hs.to(pad.dtype)
            rows.append(pad)
        hidden_states = torch.stack(rows, dim=0)         # [B, S, H*ncap]
        return DFlashTargetOutput(hidden_states=hidden_states, input_ids=input_ids,
                                  attention_mask=attention_mask, loss_mask=loss_mask)
'''

def patch_target_model():
    s = open(TM).read()
    if "OfflineDFlashTargetModel" in s:
        print("[patch] target_model уже пропатчен"); return
    # добавить класс в конец
    s += OFFLINE_CLASS
    # добавить ветку в get_dflash_target_model
    s = re.sub(
        r'(def get_dflash_target_model\([^)]*\)[^:]*:)',
        r'\1\n    import os as _os\n    if backend == "offline":\n        return OfflineDFlashTargetModel.from_pretrained(pretrained_model_name_or_path, hidden_dir=_os.environ.get("OFFLINE_HIDDEN_DIR"))',
        s, count=1)
    open(TM, "w").write(s)
    print("[patch] target_model: +OfflineDFlashTargetModel +offline-ветка")

def patch_train_args():
    # offline hidden-dir читается из ENV OFFLINE_HIDDEN_DIR (без CLI-арга) -> только +choice "offline"
    s = open(TR).read()
    if 'choices=["sglang", "hf", "offline"]' in s:
        print("[patch] train_dflash choices уже с offline"); return
    s = s.replace('choices=["sglang", "hf"]', 'choices=["sglang", "hf", "offline"]')
    open(TR, "w").write(s)
    print("[patch] train_dflash: +offline backend choice (hidden-dir через ENV OFFLINE_HIDDEN_DIR)")

if __name__ == "__main__":
    patch_target_model()
    patch_train_args()
    print("[patch] DONE")
