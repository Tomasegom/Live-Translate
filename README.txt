Working settings: (In a HP Victus Ryzen 7 7700 - RTX 4050)
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

accepted language codes: af, am, ar, as, az, ba, be, bg, bn, bo, br, bs, ca, cs, cy, da, de, el, en, es, 
et, eu, fa, fi, fo, fr, gl, gu, ha, haw, he, hi, hr, ht, hu, hy, id, is, it, ja, jw, ka, kk, km, kn, ko, 
la, lb, ln, lo, lt, lv, mg, mi, mk, ml, mn, mr, ms, mt, my, ne, nl, nn, no, oc, pa, pl, ps, pt, ro, ru, 
sa, sd, si, sk, sl, sn, so, sq, sr, su, sv, sw, ta, te, tg, th, tk, tl, tr, tt, uk, ur, uz, vi, yi, yo, zh, yue

Installs needed: (as of Aug, 2025)
    cuDNN 9 for CUDA 12: https://developer.nvidia.com/cudnn
    cuBLAS for CUDA 12 (Linux): https://developer.nvidia.com/cublas

    Using cmd
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install faster-whisper
        pip install sounddevice
        pip install numpy
        pip install keyboard
        pip install deep-translator   # (opcional, solo si quieres traducción a otros idiomas)

