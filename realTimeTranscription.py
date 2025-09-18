import sys
import threading
import queue
import sounddevice as sd
import numpy as np
import noisereduce as nr
import torch
from collections import deque

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, pyqtSignal

from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, collect_chunks


# ==============================
# CONFIGURACIÓN STT
# ==============================
samplerate = 16000
block_duration = 0.35
chunk_duration = 3
overlap_seconds = 0.5
channels = 1

use_noise_reduction = True
target_rms = 0.06
rms_threshold = 0.035

silero_threshold = 0.70
silero_min_speech_ms = 400
silero_min_silence_ms = 500

whisper_task = "translate"   # "translate" = inglés, "transcribe" = mismo idioma
whisper_lang = "es"

no_speech_prob_thresh = 0.60
unique_ratio_thresh = 0.50
dedupe_window = 4

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)
overlap_frames = int(samplerate * overlap_seconds)

audio_queue = queue.Queue()
audio_buffer = []
recent_texts = deque(maxlen=dedupe_window)


# ==============================
# MODELOS
# ==============================
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

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
    x = x.astype(np.float32)
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    gain = target / rms
    x = x * np.clip(gain, 0.25, 8.0)
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
# INTERFAZ
# ==============================
class STTApp(QWidget):
    # señal para enviar nuevo segmento a la UI (seguro desde hilos)
    new_segment = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.all_texts = []   # historial de frases

        # ==============================
        # CONFIGURACIÓN UI
        # ==============================
        FONT_SIZE = 35
        WINDOW_TITLE = "Live Translation"
        ICON_PATH = "LogoSquare.png"

        # Colores
        BG_COLOR = "#0A302E"
        TEXT_BG_COLOR = "#ffffff"
        TEXT_COLOR = "#000000"
        BTN_BG_COLOR = "#1a7439"
        BTN_TEXT_COLOR = "#ffffff"

        # color de resaltado (usamos el del botón para coherencia)
        self.HIGHLIGHT_COLOR = BTN_BG_COLOR

        # ==============================
        # VENTANA
        # ==============================
        self.setWindowTitle(WINDOW_TITLE)
        self.setWindowIcon(QIcon(ICON_PATH))
        self.showMaximized()

        layout = QVBoxLayout()

        # ==============================
        # ÁREA DE TEXTO
        # ==============================
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setFont(QFont("Tahoma", FONT_SIZE))
        self.text_area.setStyleSheet(f"""
            background-color: {TEXT_BG_COLOR};
            color: {TEXT_COLOR};
            border: 2px solid white;
            border-radius: 15px;
            padding: 15px;
        """)
        layout.addWidget(self.text_area)

        # ==============================
        # BOTÓN
        # ==============================
        self.btn_start = QPushButton("Start")
        self.btn_start.setFont(QFont("Tahoma", FONT_SIZE))
        self.btn_start.setStyleSheet(f"""
            background-color: {BTN_BG_COLOR};
            color: {BTN_TEXT_COLOR};
            border-radius: 25px;
            padding: 10px;
        """)
        self.btn_start.clicked.connect(self.toggle_stt)
        layout.addWidget(self.btn_start)

        # ==============================
        # APLICAR LAYOUT
        # ==============================
        self.setLayout(layout)
        self.setStyleSheet(f"background-color: {BG_COLOR};")

        # Variables de control
        self.running = False
        self.thread_recorder = None
        self.thread_transcriber = None

        # conectar señal a slot UI
        self.new_segment.connect(self.handle_new_segment)

    def toggle_stt(self):
        global audio_buffer
        if not self.running:
            self.running = True
            self.btn_start.setText("Stop")
            # limpia buffers/colas al iniciar
            audio_buffer = []
            self._flush_queue(audio_queue)

            self.thread_recorder = threading.Thread(target=self.recorder, daemon=True)
            self.thread_transcriber = threading.Thread(target=self.transcriber, daemon=True)
            self.thread_reccriber = self.thread_recorder  # typo guard, not used but fine
            self.thread_recorder.start()
            self.thread_transcriber.start()
        else:
            self.running = False
            self.btn_start.setText("Start")
            # opcional: vaciar cola para que se detenga rápido
            self._flush_queue(audio_queue)

    # util para limpiar colas
    def _flush_queue(self, q: queue.Queue):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    # ==============================
    # HILOS DE AUDIO
    # ==============================
    def recorder(self):
        with sd.InputStream(samplerate=samplerate, channels=channels,
                            callback=self.audio_callback, blocksize=frames_per_block):
            while self.running:
                sd.sleep(50)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print("Audio status:", status)
        audio_queue.put(indata.copy())

    def transcriber(self):
        global audio_buffer, recent_texts
        while self.running:
            try:
                block = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            audio_buffer.append(block)
            total_frames = sum(len(b) for b in audio_buffer)

            if total_frames >= frames_per_chunk:
                # concatenar y mantener overlap
                audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
                audio_buffer = [audio_data[-overlap_frames:]]
                audio_data = audio_data.flatten().astype(np.float32)

                # limpieza
                audio_data = remove_dc_and_normalize(audio_data, target=target_rms)
                audio_data = apply_noise_reduction(audio_data)

                # VAD
                wav_tensor = torch.from_numpy(audio_data)
                timestamps = get_speech_timestamps(
                    wav_tensor, vad_model, sampling_rate=samplerate,
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
                if not is_voice(speech_np):
                    continue
                if len(speech_np) < int(0.5 * samplerate):
                    continue

                # Whisper
                segments, _ = model.transcribe(
                    speech_np,
                    task=whisper_task,
                    language=whisper_lang,
                    beam_size=4,
                    vad_filter=False
                )

                for seg in segments:
                    if not self.running:
                        break

                    text = seg.text.strip()
                    # filtros
                    if seg.no_speech_prob and seg.no_speech_prob > no_speech_prob_thresh:
                        continue
                    words = text.split()
                    if len(words) < 2:
                        continue
                    unique_ratio = len(set(words)) / (len(words) + 1e-9)
                    if unique_ratio < unique_ratio_thresh:
                        continue
                    if text.lower() in (t.lower() for t in recent_texts):
                        continue

                    recent_texts.append(text)
                    # emitir a la UI de forma segura
                    self.new_segment.emit(text)

    # ==============================
    # SLOT: Manejar nuevo segmento (UI thread)
    # ==============================
    def handle_new_segment(self, text: str):
        # guardar historial
        self.all_texts.append(text)

        # anteriores → se apilan arriba (más chicos y grises)
        previous_html = ""
        for t in self.all_texts[:-1]:
            previous_html += f"<p style='color:#888888; font-size:25px; margin:2px 0;'>{t}</p>"

        # último → siempre fijo en el centro
        last_html = f"""
        <div style='display:flex; justify-content:center; align-items:center; height:100%;'>
            <p style='color:{self.HIGHLIGHT_COLOR}; font-weight:bold; font-size:50px; text-align:center;'>
                {self.all_texts[-1]}
            </p>
        </div>
        """

        # construir el contenido final
        html_content = f"""
        <div style='height:35%; overflow-y:auto;'>
            {previous_html}
        </div>
        <div style='height:65%; display:flex; justify-content:center; align-items:center;'>
            {last_html}
        </div>
        """

        self.text_area.setHtml(html_content)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = STTApp()
    window.show()
    sys.exit(app.exec_())
