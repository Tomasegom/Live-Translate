import sys
import threading
import queue
import sounddevice as sd
import numpy as np
import noisereduce as nr
import torch
from collections import deque

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QFileDialog
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, collect_chunks

# CODE TAKES APPROXIMATELY ~17 SECONDS TO LAUNCH

# ==============================
# CONFIGURACIÓN STT
# ==============================
samplerate = 16000
block_duration = 0.35
chunk_duration = 3
overlap_seconds = 0.5
channels = 1

use_noise_reduction = True
target_rms = 0.08
rms_threshold = 0.065

silero_threshold = 0.70
silero_min_speech_ms = 750
silero_min_silence_ms = 500

whisper_task = "translate"   # "translate" = inglés, "transcribe" = mismo idioma
whisper_lang = "es"

no_speech_prob_thresh = 0.75
unique_ratio_thresh = 0.50
dedupe_window = 3

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
    new_segment = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.all_texts = []   # historial de frases

        # ==============================
        # CONFIGURACIÓN UI
        # ==============================
        FONT_SIZE = 25
        WINDOW_TITLE = "Live Translation"
        ICON_PATH = "LogoSquare.png"

        # Colores
        BG_COLOR = "#0A302E"
        TEXT_BG_COLOR = "#ffffff"
        TEXT_COLOR = "#000000"
        BTN_BG_COLOR = "#1a7439"
        BTN_TEXT_COLOR = "#ffffff"

        self.HIGHLIGHT_COLOR = BTN_BG_COLOR

        # ==============================
        # VENTANA
        # ==============================
        self.setWindowTitle(WINDOW_TITLE)
        self.setWindowIcon(QIcon(ICON_PATH))
        self.showMaximized()

        layout = QVBoxLayout()

        # ==============================
        # HISTORIAL (arriba, scrollable)
        # ==============================
        self.history_area = QTextEdit()
        self.history_area.setReadOnly(True)
        self.history_area.setFont(QFont("Tahoma", 25))
        self.history_area.setStyleSheet(f"""
            background-color: {TEXT_BG_COLOR};
            color: #000000;
            border: 2px solid white;
            border-radius: 10px;
            padding: 10px;
        """)
        layout.addWidget(self.history_area, stretch=2)   # el 2 da más espacio al historial

        # ==============================
        # TEXTO RESALTADO (caja blanca, justo encima del botón)
        # ==============================
        self.highlight_area = QTextEdit()
        self.highlight_area.setReadOnly(True)
        self.highlight_area.setFont(QFont("Tahoma", 90, QFont.Bold))
        self.highlight_area.setPlaceholderText("Live Translation")
        self.highlight_area.setStyleSheet(f"""
            background-color: {TEXT_BG_COLOR};
            color: {self.HIGHLIGHT_COLOR};
            border: 2px solid white;
            border-radius: 10px;
            padding: 15px;
        """)
        layout.addWidget(self.highlight_area, stretch=1)  # ocupa más espacio dinámico que el botón

        # ==============================
        # BOTÓN START/STOP
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
        layout.addWidget(self.btn_start, stretch=0)   # el botón no crece

        # ==============================
        # BOTÓN GUARDAR HISTORIAL
        # ==============================
        self.btn_save = QPushButton("Save")
        self.btn_save.setFont(QFont("Tahoma", FONT_SIZE))
        self.btn_save.setStyleSheet("""
            background-color: #1177bb;
            color: white;
            border-radius: 25px;
            padding: 10px;
        """)
        self.btn_save.clicked.connect(self.save_history_to_file)
        layout.addWidget(self.btn_save, stretch=0)


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
        
        # === Calibración automática ===
        self.noise_samples = deque(maxlen=200)   # guarda RMS de silencios recientes
        self.calib_timer = QTimer(self)
        self.calib_timer.setInterval(60_000)     # 60 segundos
        self.calib_timer.timeout.connect(self.auto_recalibrate)
        self.calib_timer.start()

        
    def save_history_to_file(self): #Para guardar el historial en un .txt
        if not self.all_texts:
            return  # si no hay nada, no guardamos

        # abrir diálogo para elegir archivo
        filename, _ = QFileDialog.getSaveFileName(
            self, "Guardar historial", "historial.txt", "Text Files (*.txt)"
        )
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                for line in self.all_texts:
                    f.write(line + "\n")

    def toggle_stt(self):
        global audio_buffer
        if not self.running:
            self.running = True
            self.btn_start.setText("Stop")
            audio_buffer = []
            self._flush_queue(audio_queue)

            self.thread_recorder = threading.Thread(target=self.recorder, daemon=True)
            self.thread_transcriber = threading.Thread(target=self.transcriber, daemon=True)
            self.thread_recorder.start()
            self.thread_transcriber.start()
        else:
            self.running = False
            self.btn_start.setText("Start")
            self._flush_queue(audio_queue)

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
                audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
                audio_buffer = [audio_data[-overlap_frames:]]
                audio_data = audio_data.flatten().astype(np.float32)
                
                raw_rms = float(np.sqrt(np.mean(audio_data**2))) + 1e-12

                audio_data = remove_dc_and_normalize(audio_data, target=target_rms)
                audio_data = apply_noise_reduction(audio_data)

                wav_tensor = torch.from_numpy(audio_data)
                timestamps = get_speech_timestamps(
                    wav_tensor, vad_model, sampling_rate=samplerate,
                    threshold=silero_threshold,
                    min_speech_duration_ms=silero_min_speech_ms,
                    min_silence_duration_ms=silero_min_silence_ms
                )
                if not timestamps:
                    self.noise_samples.append(raw_rms)
                    continue
                speech_tensor = collect_chunks(timestamps, wav_tensor)
                
                if speech_tensor.numel() == 0:
                    self.noise_samples.append(raw_rms)
                    continue
                speech_np = speech_tensor.numpy().astype(np.float32)
                speech_np = remove_dc_and_normalize(speech_np, target=target_rms)
                
                if not is_voice(speech_np):
                    self.noise_samples.append(raw_rms)
                    continue
                
                if len(speech_np) < int(0.5 * samplerate):
                    self.noise_samples.append(raw_rms)
                    continue

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
                    self.new_segment.emit(text)

    # ==============================
    # SLOT UI
    # ==============================
    def handle_new_segment(self, text: str):
        self.all_texts.append(text)

        # historial arriba
        history_html = ""
        for t in self.all_texts[:-1]:
            history_html += f"<p style='margin:2px 0;'>{t}</p>"
        self.history_area.setHtml(history_html)

        # auto-scroll al final
        cursor = self.history_area.textCursor()
        cursor.movePosition(cursor.End)
        self.history_area.setTextCursor(cursor)
        self.history_area.ensureCursorVisible()

        # texto resaltado en caja blanca dinámica
        self.highlight_area.setHtml(
            f"<p style='color:{self.HIGHLIGHT_COLOR}; font-weight:bold; font-size:90px; text-align:center;'>{self.all_texts[-1]}</p>"
        )
    
    def auto_recalibrate(self):
        """Ajusta umbrales en función del ruido mediano reciente."""
        global rms_threshold, target_rms, no_speech_prob_thresh

        if len(self.noise_samples) < 10:
            return  # no hay suficientes samples

        noise_med = float(np.median(self.noise_samples))

        # Calcula nuevos valores con límites seguros
        new_rms_threshold = np.clip(noise_med * 3.0, 0.015, 0.08)
        new_target_rms    = np.clip(max(0.06, noise_med * 5.0), 0.06, 0.14)

        # Ayuda a Whisper en ambientes más ruidosos
        new_no_speech = 0.75 if noise_med < 0.02 else 0.80

        rms_threshold = float(new_rms_threshold)
        target_rms = float(new_target_rms)
        no_speech_prob_thresh = float(new_no_speech)

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = STTApp()
    window.showMaximized()
    sys.exit(app.exec_())
