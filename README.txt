Tested succesfully for 50 minutes straight with youtube video

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

Working settings: (In a HP Victus Ryzen 7 7700 - RTX 4050)
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

FASTER WHISPER (LISTEN ONLY) accepted language codes: 
    af, am, ar, as, az, ba, be, bg, bn, bo, br, bs, ca, cs, cy, da, de, el, en, es, 
    et, eu, fa, fi, fo, fr, gl, gu, ha, haw, he, hi, hr, ht, hu, hy, id, is, it, ja, jw, ka, kk, km, kn, ko, 
    la, lb, ln, lo, lt, lv, mg, mi, mk, ml, mn, mr, ms, mt, my, ne, nl, nn, no, oc, pa, pl, ps, pt, ro, ru, 
    sa, sd, si, sk, sl, sn, so, sq, sr, su, sv, sw, ta, te, tg, th, tk, tl, tr, tt, uk, ur, uz, vi, yi, yo, zh, yue

DEEP_TRANSLATE - GOOGLE TRANSLATOR (Language list):
    afrikaans : af
    albanian : sq
    amharic : am
    arabic : ar
    armenian : hy
    assamese : as
    aymara : ay
    azerbaijani : az
    bambara : bm
    basque : eu
    belarusian : be
    bengali : bn
    bhojpuri : bho
    bosnian : bs
    bulgarian : bg
    catalan : ca
    cebuano : ceb
    chichewa : ny
    chinese (simplified) : zh-CN
    chinese (traditional) : zh-TW
    corsican : co
    croatian : hr
    czech : cs
    danish : da
    dhivehi : dv
    dogri : doi
    dutch : nl
    english : en
    esperanto : eo
    estonian : et
    ewe : ee
    filipino : tl
    finnish : fi
    french : fr
    frisian : fy
    galician : gl
    georgian : ka
    german : de
    greek : el
    guarani : gn
    gujarati : gu
    haitian creole : ht
    hausa : ha
    hawaiian : haw
    hebrew : iw
    hindi : hi
    hmong : hmn
    hungarian : hu
    icelandic : is
    igbo : ig
    ilocano : ilo
    indonesian : id
    irish : ga
    italian : it
    japanese : ja
    javanese : jw
    kannada : kn
    kazakh : kk
    khmer : km
    kinyarwanda : rw
    konkani : gom
    korean : ko
    krio : kri
    kurdish (kurmanji) : ku
    kurdish (sorani) : ckb
    kyrgyz : ky
    lao : lo
    latin : la
    latvian : lv
    lingala : ln
    lithuanian : lt
    luganda : lg
    luxembourgish : lb
    macedonian : mk
    maithili : mai
    malagasy : mg
    malay : ms
    malayalam : ml
    maltese : mt
    maori : mi
    marathi : mr
    meiteilon (manipuri) : mni-Mtei
    mizo : lus
    mongolian : mn
    myanmar : my
    nepali : ne
    norwegian : no
    odia (oriya) : or
    oromo : om
    pashto : ps
    persian : fa
    polish : pl
    portuguese : pt
    punjabi : pa
    quechua : qu
    romanian : ro
    russian : ru
    samoan : sm
    sanskrit : sa
    scots gaelic : gd
    sepedi : nso
    serbian : sr
    sesotho : st
    shona : sn
    sindhi : sd
    sinhala : si
    slovak : sk
    slovenian : sl
    somali : so
    spanish : es
    sundanese : su
    swahili : sw
    swedish : sv
    tajik : tg
    tamil : ta
    tatar : tt
    telugu : te
    thai : th
    tigrinya : ti
    tsonga : ts
    turkish : tr
    turkmen : tk
    twi : ak
    ukrainian : uk
    urdu : ur
    uyghur : ug
    uzbek : uz
    vietnamese : vi
    welsh : cy
    xhosa : xh
    yiddish : yi
    yoruba : yo
    zulu : zu