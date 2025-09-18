import sys
import threading
import time
from PyQt5 import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit
from PyQt5.QtGui import QFont, QIcon

class STTApp(QWidget):
    def __init__(self):
        super().__init__()

        # ==============================
        # CONFIGURACIÓN
        # ==============================
        FONT_SIZE = 35
        WINDOW_TITLE = "Live Translation"
        ICON_PATH = "LogoSquare.png"   # icono .png o .ico

        # Colores (fácil de cambiar)
        BG_COLOR = "#0A302E"         # Fondo de la ventana
        TEXT_BG_COLOR = "#ffffff"     # Fondo del área de texto
        TEXT_COLOR = "#000000"        # Color de texto
        BTN_BG_COLOR = "#1a7439"      # Fondo botón
        BTN_TEXT_COLOR = "#ffffff"    # Texto botón

        # ==============================
        # VENTANA
        # ==============================
        self.setWindowTitle(WINDOW_TITLE)
        self.setWindowIcon(QIcon(ICON_PATH))   # Icono ventana
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
        # APLICAR LAYOUT + FONDO VENTANA
        # ==============================
        self.setLayout(layout)
        self.setStyleSheet(f"background-color: {BG_COLOR};")

        # Variables de control
        self.running = False
        self.thread = None

    def toggle_stt(self):
        """Inicia o detiene el hilo de STT"""
        if not self.running:
            self.running = True
            self.btn_start.setText("Stop")
            self.thread = threading.Thread(target=self.fake_stt, daemon=True)
            self.thread.start()
        else:
            self.running = False
            self.btn_start.setText("Start")

    def fake_stt(self):
        """Simula resultados en vivo (ejemplo, luego reemplazamos con Whisper)"""
        counter = 1
        while self.running:
            text = f"Reconocido fragmento {counter}\n"
            self.text_area.append(text)
            counter += 1
            time.sleep(1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = STTApp()
    window.show()
    sys.exit(app.exec_())
