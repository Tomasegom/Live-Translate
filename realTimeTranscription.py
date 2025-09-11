import torchaudio
import torch
import sounddevice as sd
import numpy as np
import queue
import threading
import keyboard
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, collect_chunks

# ==============================
# CONFIGURACIÓN
# ==============================
samplerate = 16000
block_duration = 0.5   # tamaño de cada bloque (segundos)
chunk_duration = 3     # duración de cada chunk a transcribir, tiempo en llenar (segundos)
channels = 1           # 1: mono, 2: stereo (Para live translation es mejor 1)

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []
stop_flag = False       # bandera global -> se pone en True al presionar Q
last_text = ""          # evita repeticiones seguidas

# ==============================
# MODELOS
# ==============================
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


# ==============================
# CALLBACK DE AUDIO
# ==============================
def audio_callback(indata, frames, time, status):
    if status:
        print("Error:", status)
    audio_queue.put(indata.copy())


# ==============================
# RECORDER: toma audio del micro
# ==============================
def recorder():
    global stop_flag
    with sd.InputStream(samplerate=samplerate, channels=channels,
                        callback=audio_callback, blocksize=frames_per_block):
        print("Listening... (press Q to stop)")
        while not stop_flag:   # mientras no se haya presionado Q
            sd.sleep(100)


# ==============================
# TRANSCRIBER: procesa audio y transcribe
# ==============================
def transcriber():
    global audio_buffer, stop_flag, last_text
    while not stop_flag:   # sale cuando se presiona Q
        try:
            block = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        audio_buffer.append(block)
        total_frames = sum(len(b) for b in audio_buffer)

        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            overlap = int(0.5 * samplerate)
            audio_buffer = [audio_data[-overlap:]]  # mantiene un poco de contexto

            audio_data = audio_data.flatten().astype(np.float32)

            # VAD (detección de voz con Silero)
            wav_tensor = torch.from_numpy(audio_data)
            timestamps = get_speech_timestamps(
                wav_tensor,
                vad_model,
                sampling_rate=samplerate,
                threshold=0.7,
                min_speech_duration_ms=400, #evita pedazos cortos
                min_silence_duration_ms=250 #hay silencio entre frases
            )

            def is_voice(audio, threshold=0.04):
                energy = np.sqrt(np.mean(audio**2))
                return energy > threshold

            if not timestamps:
                continue

            speech_tensor = collect_chunks(timestamps, wav_tensor)
            if speech_tensor.numel() == 0:
                continue

            speech_np = speech_tensor.numpy().astype(np.float32)
            speech_np /= np.max(np.abs(speech_np)) + 1e-9

            if len(speech_np) < 0.5 * samplerate:
                continue

            # Transcripción
            segments, _ = model.transcribe(
                speech_np,
                task="translate",   # traduce a inglés
                language="es",      # audio original en español
                beam_size=5,
                vad_filter=False
            )

            for segment in segments:
                text = segment.text.strip()

                # Filtros de calidad
                if len(text) < 2:
                    print("[...]")
                    continue
                if segment.no_speech_prob > 0.6: # 0 - 1 Entre mas alto, menos ruido pasa
                    print("[...]")
                    continue
                if text.lower() == last_text.lower():
                    continue
                if not is_voice(speech_np, threshold=0.02): # RMS threshold for voice (0.02 - 0.1)
                    continue
                #Filtro por eco (si se repite muchas veces una palabra)(bocinas repitiendo)
                unique_ratio = len(set(text.split())) / (len(text.split()) + 1e-9)
                if unique_ratio < 0.5:   # demasiada repetición -> probable eco/ruido
                    continue

                # Mostrar texto válido
                print(text)
                last_text = text


# ==============================
# KEY LISTENER: espera la tecla Q
# ==============================
def key_listener():
    global stop_flag
    keyboard.wait("q")       # se queda esperando
    stop_flag = True         # al presionar Q, todo se detiene
    print("\n Program finished.")


# ==============================
# INICIO DE THREADS
# ==============================
threading.Thread(target=recorder, daemon=True).start()
threading.Thread(target=key_listener, daemon=True).start()
transcriber()  # corre en el hilo principal
