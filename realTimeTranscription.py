import torchaudio
import torch
import sounddevice as sd
import numpy as np
import queue
import threading
import keyboard
import noisereduce as nr
from collections import deque
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, collect_chunks

# ==============================
# CONFIGURACIÓN (tunea aquí)
# ==============================
samplerate = 16000

# Captura y segmentación
block_duration = 0.35        # s (bloques del mic)
chunk_duration = 3           # s (ventana a transcribir)
overlap_seconds = 0.5        # s (contexto para no cortar frases)
channels = 1

# Limpieza de audio
use_noise_reduction = True   # activar/desactivar noisereduce
target_rms = 0.06            # normalización RMS objetivo (~ -26 dBFS)

# VAD Silero
silero_threshold = 0.70         # 0.6–0.75 (más alto = más estricto)
silero_min_speech_ms = 400      # mínimo de voz 
silero_min_silence_ms = 500     # silencio para marcar fin de frase 

# Whisper
whisper_task = "translate"   # "translate" => SIEMPRE inglés; "transcribe" => mismo idioma
whisper_lang = "es"          # idioma del audio de entrada

# Filtros de texto y audio
no_speech_prob_thresh = 0.60
unique_ratio_thresh = 0.50   # repetición de palabras
dedupe_window = 4            # recuerda N últimas salidas para evitar repetidos
rms_threshold = 0.035        

# ==============================
# DERIVADOS
# ==============================
frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)
overlap_frames = int(samplerate * overlap_seconds)

audio_queue = queue.Queue()
audio_buffer = []
stop_flag = False
recent_texts = deque(maxlen=dedupe_window)

# ==============================
# MODELOS
# ==============================
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
#model = WhisperModel("medium", device="cuda", compute_type="float16")

# Silero
vad_model, utils = torch.hub.load(
    "snakers4/silero-vad",
    "silero_vad",
    trust_repo=True,
    force_reload=False,
    skip_validation=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# ==============================
# UTILIDADES AUDIO
# ==============================
def remove_dc_and_normalize(x: np.ndarray, target=target_rms):
    # Quita DC-offset y normaliza a RMS objetivo
    x = x.astype(np.float32)
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    gain = target / rms
    x = x * np.clip(gain, 0.25, 8.0)  # evita explosiones (ganancia 0.25–8x)
    x = np.clip(x, -1.0, 1.0)
    return x

def apply_noise_reduction(x: np.ndarray):
    if not use_noise_reduction:
        return x
    return nr.reduce_noise(y=x, sr=samplerate)

def is_voice(x: np.ndarray, threshold=rms_threshold):
    energy = np.sqrt(np.mean(x**2))
    return energy > threshold

# ==============================
# CALLBACK DE AUDIO
# ==============================
def audio_callback(indata, frames, time, status):
    if status:
        print("Audio status:", status)
    audio_queue.put(indata.copy())

# ==============================
# HILO DE CAPTURA
# ==============================
def recorder():
    global stop_flag
    with sd.InputStream(samplerate=samplerate, channels=channels,
                        callback=audio_callback, blocksize=frames_per_block):
        print("Listening… (press Q to stop)")
        while not stop_flag:
            sd.sleep(50)

# ==============================
# TRANSCRIPCIÓN
# ==============================
def transcriber():
    global audio_buffer, stop_flag, recent_texts
    while not stop_flag:
        try:
            block = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        audio_buffer.append(block)
        total_frames = sum(len(b) for b in audio_buffer)

        if total_frames >= frames_per_chunk:
            # Concat + overlap para contexto
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            audio_buffer = [audio_data[-overlap_frames:]]  # conserva cola

            # Aplanar → limpieza
            audio_data = audio_data.flatten().astype(np.float32)
            audio_data = remove_dc_and_normalize(audio_data, target=target_rms)
            audio_data = apply_noise_reduction(audio_data)

            # ===== VAD: Silero =====
            wav_tensor = torch.from_numpy(audio_data)
            timestamps = get_speech_timestamps(
                wav_tensor,
                vad_model,
                sampling_rate=samplerate,
                threshold=silero_threshold,
                min_speech_duration_ms=silero_min_speech_ms,
                min_silence_duration_ms=silero_min_silence_ms
            )
            if not timestamps:
                continue

            speech_tensor = collect_chunks(timestamps, wav_tensor)
            if speech_tensor.numel() == 0:
                continue

            speech_np = speech_tensor.numpy().astype(np.float32)
            speech_np = remove_dc_and_normalize(speech_np, target=target_rms)

            # RMS filter
            if not is_voice(speech_np):
                continue

            # Longitud mínima útil
            if len(speech_np) < int(0.5 * samplerate):
                continue

            # ===== Whisper =====
            segments, _ = model.transcribe(
                speech_np,
                task=whisper_task,
                language=whisper_lang,
                beam_size=4,
                vad_filter=False
            )

            for seg in segments:
                text = seg.text.strip()

                # ===== Filtros de salida =====
                if seg.no_speech_prob is not None and seg.no_speech_prob > no_speech_prob_thresh:
                    continue
                if len(text.split()) < 2:
                    continue
                words = text.split()
                unique_ratio = len(set(words)) / (len(words) + 1e-9)
                if unique_ratio < unique_ratio_thresh:
                    continue
                if text.lower() in (t.lower() for t in recent_texts):
                    continue

                print(text)
                recent_texts.append(text)

# ==============================
# HILO DE TECLA
# ==============================
def key_listener():
    global stop_flag
    keyboard.wait("q")
    stop_flag = True
    print("\nProgram finished.")

# ==============================
# ARRANQUE
# ==============================
threading.Thread(target=recorder, daemon=True).start()
threading.Thread(target=key_listener, daemon=True).start()
transcriber()
