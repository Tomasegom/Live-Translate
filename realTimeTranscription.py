import torchaudio
import torch
import sounddevice as sd
import numpy as np
import queue
import threading
import keyboard
import noisereduce as nr
import webrtcvad
from collections import deque
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, collect_chunks

# ==============================
# CONFIGURACIÓN (tunea aquí)
# ==============================
samplerate = 16000

# Captura y segmentación
block_duration = 0.35        # s (bloques del mic)
chunk_duration = 2.5         # s (ventana a transcribir)
overlap_seconds = 0.7        # s (contexto para no cortar frases)
channels = 1

# Limpieza de audio
use_noise_reduction = True   # activar/desactivar noisereduce
target_rms = 0.055            # ~ -26 dBFS aprox (0.03-0.07 recomendado)

# VADs
silero_threshold = 0.70      # 0.6–0.75 (más alto = más estricto)
silero_min_speech_ms = 400
silero_min_silence_ms = 250

webrtc_aggressiveness = 2    # 0 permisivo .. 3 estricto
webrtc_frame_ms = 20         # 10/20/30 ms válidos
webrtc_min_voiced_ratio = 0.5 # % de frames con voz dentro del chunk (0.3–0.6)

# Whisper
whisper_task = "translate"   # "translate" => SIEMPRE inglés; "transcribe" => mismo idioma
whisper_lang = "es"          # idioma del audio de entrada

# Filtros de texto
no_speech_prob_thresh = 0.60
unique_ratio_thresh = 0.50   # repetición de palabras
#blacklist = {"i don't know", "thank you", "ok", "okay", "gracias", "mmm", "uh"}
dedupe_window = 3            # recuerda N últimas salidas para evitar repetidos

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

# Silero
vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# WebRTC
webrtc_vad = webrtcvad.Vad(webrtc_aggressiveness)

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
    # noisereduce asume float32 [-1,1]; usar con cuidado (latencia extra)
    return nr.reduce_noise(y=x, sr=samplerate)

def webrtc_voiced_ratio(x: np.ndarray) -> float:
    """Evalúa varios frames consecutivos con WebRTC VAD y retorna la fracción con voz."""
    # WebRTC VAD requiere PCM16 mono a 8/16/32/48 kHz y frames 10/20/30 ms
    pcm16 = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    bytes_per_frame = int(samplerate * webrtc_frame_ms / 1000) * 2  # 2 bytes (int16)
    if len(pcm16) < bytes_per_frame:
        return 0.0
    n_frames = len(pcm16) // bytes_per_frame
    voiced = 0
    for i in range(n_frames):
        frame = pcm16[i * bytes_per_frame : (i + 1) * bytes_per_frame]
        try:
            if webrtc_vad.is_speech(frame, samplerate):
                voiced += 1
        except Exception:
            # Si falla por frame inválido, lo ignoramos
            pass
    return voiced / max(1, n_frames)

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
        print(f"Listening… (press Q to stop)")
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
            # Concat, overlap para contexto
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            audio_buffer = [audio_data[-overlap_frames:]]  # conserva cola

            # Aplanar → limpieza → (opcional) NR
            audio_data = audio_data.flatten().astype(np.float32)
            audio_data = remove_dc_and_normalize(audio_data, target=target_rms)
            audio_data = apply_noise_reduction(audio_data)

            # ===== VAD 1: Silero =====
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
            # Normaliza cada recorte de voz también (ayuda con volumen variable)
            speech_np = remove_dc_and_normalize(speech_np, target=target_rms)

            # ===== VAD 2: WebRTC (multi-frame) =====
            voiced_ratio = webrtc_voiced_ratio(speech_np)
            if voiced_ratio < webrtc_min_voiced_ratio:
                continue

            # Longitud mínima útil (evitar aplausos o chasquidos)
            if len(speech_np) < int(0.5 * samplerate):
                continue

            # ===== Whisper =====
            segments, _ = model.transcribe(
                speech_np,
                task=whisper_task,
                language=whisper_lang,
                beam_size=4,
                vad_filter=False  # usamos Silero+WebRTC
            )

            for seg in segments:
                text = seg.text.strip()

                # ===== Filtros de salida =====
                # 1) confianza de "no speech"
                if seg.no_speech_prob is not None and seg.no_speech_prob > no_speech_prob_thresh:
                    continue

                # 2) palabras mínimas
                if len(text.split()) < 2:
                    continue

                # 3) blacklist de muletillas comunes
                '''if text.lower() in blacklist:
                    continue'''

                # 4) repetición (eco/loop por bocinas)
                words = text.split()
                unique_ratio = len(set(words)) / (len(words) + 1e-9)
                if unique_ratio < unique_ratio_thresh:
                    continue

                # 5) dedupe por ventana reciente
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
