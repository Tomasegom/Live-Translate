Installs needed: (as of SEPT, 2025)
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

pip install...
########################################
# 📦 CORE - Dependencias principales
# Necesarias para que funcione el sistema básico
########################################
    torch
    torchaudio
    sounddevice
    numpy
    keyboard
    faster-whisper
    silero-vad
    PyQT5 --> UI

########################################
# 🔊 PREPROCESAMIENTO - Opcionales
# Mejoran la robustez en ambientes ruidosos
########################################
    noisereduce
    webrtcvad
    pydub
    torch-audiomentations

########################################
# 📝 NLP & POST-PROCESAMIENTO - Opcionales
# Mejoran la calidad del texto generado
########################################
    transformers
    nltk
    spacy

########################################
# 📊 MÉTRICAS - Opcionales
# Para evaluar precisión de transcripción (WER, etc.)
########################################
    jiwer

########################################
# 🗣️ DIARIZACIÓN - Opcional (pesado)
# Identificación de hablantes múltiples
########################################
    pyannote.audio
