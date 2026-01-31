import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import ipywidgets as widgets
from IPython.display import display
import os

class VoiceRecorder:
    def __init__(self, output_area, openai_client, wav_path=None, txt_path=None, sample_rate=44100):
        """
        Inicjalizacja rejestratora dźwięku.
        
        Args:
            output_area: Widget Output z ipywidgets do wyświetlania komunikatów.
            openai_client: Zainicjalizowany klient OpenAI.
            wav_path (str): Ścieżka do zapisu pliku .wav.
            sample_rate (int): Częstotliwość próbkowania (domyślnie 44100).
            txt_path (str): Ścieżka do zapisu pliku .txt z transkrypcją.
        """
        self.output_area = output_area
        self.client = openai_client
        self.sample_rate = sample_rate
        self.txt_path = txt_path
        
        # Ustawienie domyślnej ścieżki
        if wav_path:
            self.wav_path = wav_path
        else:
            self.wav_path = r"voice\input.wav"

        # Zmienne stanu
        self.is_recording = False
        self.audio_buffer = []
        self.stream = None
        self.transcribed_text = ""

    def _callback(self, indata, frames, time, status):
        """Wewnętrzna funkcja callback dla sounddevice."""
        if self.is_recording:
            self.audio_buffer.extend(indata[:, 0])

    def start_recording(self):
        """Rozpoczyna nagrywanie."""
        if self.is_recording:
            print("Nagrywanie już trwa.")
            return

        self.audio_buffer = []  
        self.is_recording = True
        
        # Inicjalizacja strumienia
        self.stream = sd.InputStream(
            callback=self._callback, 
            channels=1, 
            samplerate=self.sample_rate
        )
        self.stream.start()
        
        print("Nagrywanie rozpoczęte...")
        with self.output_area:
            display(widgets.HTML(f'<b>Nagrywanie rozpoczęte...</b>'))

    def stop_recording(self):
        """Zatrzymuje nagrywanie i zapisuje plik WAV."""
        if not self.is_recording:
            print("Nie ma aktywnego nagrywania.")
            return

        self.is_recording = False
        self.stream.stop()
        self.stream.close() 
        
        with self.output_area:
            display(widgets.HTML(f'<b>Nagrywanie zakończone.</b>'))

        # Konwersja i zapis
        try:
            recorded_audio = np.array(self.audio_buffer, dtype=np.float32)
            
            os.makedirs(os.path.dirname(self.wav_path), exist_ok=True)
            
            write(self.wav_path, self.sample_rate, recorded_audio)
            print(f"Plik WAV zapisany jako {self.wav_path}")
        except Exception as e:
            print(f"Błąd podczas zapisu pliku: {e}")

    def transcribe(self):
        """Wysyła zapisany plik do Whisper API."""
        if not os.path.exists(self.wav_path):
            print("Brak pliku audio do transkrypcji. Nagraj coś najpierw.")
            return

        whisper_prompt = "jesteś asystentem, który pomaga klientom w centrum handlowym. Zamieniaj mowę na tekst nagranie jest pytaniem klienta o produkt lub usługę dostępną w centrum handlowym. Nagranie jest w języku polskim."

        try:
            with open(self.wav_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    language="pl",
                    prompt=whisper_prompt
                )
            
            self.transcribed_text = transcription.text

            try:
                with open(self.txt_path, "w", encoding="utf-8") as f:
                    f.write(self.transcribed_text)
                
                msg = f"Zapisano transkrypcję do: {self.txt_path}"
            except Exception as e:
                msg = f"Tekst odczytany, ale błąd zapisu do pliku TXT: {e}"
            
            with self.output_area:
                display(widgets.HTML(f'<b>Transkrypcja:</b> {self.transcribed_text}'))
            
            return self.transcribed_text
            
        except Exception as e:
            with self.output_area:
                print(f"Błąd transkrypcji: {e}")