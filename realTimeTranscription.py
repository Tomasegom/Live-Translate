import sounddevice as sd
import numpy as np
import queue
import threading
import keyboard
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

'''
In this branch I'll test including some external translators so more languages are available
'''

# Settings
samplerate = 16000
block_duration = 0.5  # Seconds
chunk_duration = 2    # Seconds
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []
stop_flag = False  # Variable global de control

# Model Setup
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())
    
def recorder():
    global stop_flag
    with sd.InputStream(samplerate=samplerate, channels=channels, 
                        callback=audio_callback, blocksize=frames_per_block):
        print("Listening... (press Q to Stop)")
        while not stop_flag:
            sd.sleep(100)

def transcriber():
    global audio_buffer, stop_flag
    while not stop_flag:
        try:
            block = audio_queue.get(timeout=0.1)  # evita bloqueo infinito
        except queue.Empty:
            continue

        audio_buffer.append(block)
        total_frames = sum(len(b) for b in audio_buffer)

        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            audio_buffer = []  # Clears buffer

            audio_data = audio_data.flatten().astype(np.float32)

            # Transcription con VAD activado
            segments, _ = model.transcribe(
                audio_data,
                task="translate",   # Traduce a inglés - Faster Whisper nativamente solo traduce a ingles
                language="es",      # Audio original en español
                beam_size=1,
                vad_filter=True
            )


            for segment in segments:
                print(f"{segment.text}")


def key_listener():
    global stop_flag
    keyboard.wait("q")   # espera a que presiones "q"
    stop_flag = True
    print("\nPrograma terminado por el usuario.")

# Start threads
threading.Thread(target=recorder, daemon=True).start()
threading.Thread(target=key_listener, daemon=True).start()
transcriber()
