"""
Laser Microphone Suite - Advanced Audio Processing Application
Enhanced Features:
- 48kHz, 16-bit, Mono WAV recording
- Real-time bandpass filtering (300-3400 Hz)
- Live listening with filtering and volume control (1-10x amplification)
- Audio visualization (waveform, spectrogram)
- Multiple Vosk model support with selection
- Side-by-side transcription (Vosk + Speech Recognition + Whisper)
- Automatic 16kHz conversion for transcription
- Recording upload functionality
- Working playback feature with timeline
- Recording history management
- Export functionality
"""

import sys
import os
import json
import traceback
import subprocess


def is_ffmpeg_available(exec_path=None):
    """Return True if ffmpeg is callable.

    If exec_path is provided and points to an executable, that will be used.
    Otherwise the function attempts to call 'ffmpeg' on PATH.
    """
    try:
        if exec_path:
            # If provided a directory, look for ffmpeg.exe inside it
            if os.path.isdir(exec_path):
                cand = os.path.join(exec_path, 'ffmpeg.exe')
            else:
                cand = exec_path
            if not os.path.exists(cand):
                return False
            result = subprocess.run([cand, '-version'], capture_output=True, timeout=5)
            return result.returncode == 0
        else:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
            return result.returncode == 0
    except Exception:
        return False

# Check PyQt6 import first
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                  QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                                  QListWidget, QTabWidget, QComboBox, QSlider,
                                  QCheckBox, QTextEdit, QFileDialog, QMessageBox,
                                  QProgressBar, QSpinBox, QGroupBox, QListWidgetItem)
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QUrl
    from PyQt6.QtGui import QFont, QColor
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    print("✓ PyQt6 loaded successfully")
except ImportError as e:
    print(f"ERROR: PyQt6 not found. Install with: pip install PyQt6")
    print(f"Details: {e}")
    sys.exit(1)

import numpy as np
from datetime import datetime
import shutil

# Optional imports
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    print("✓ PyAudio loaded successfully")
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("WARNING: PyAudio not available. Install with: pip install pyaudio")

try:
    import wave
    WAVE_AVAILABLE = True
except ImportError:
    WAVE_AVAILABLE = False
    print("WARNING: wave module not available")

try:
    from scipy import signal
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
    print("✓ SciPy loaded successfully")
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: SciPy not available. Install with: pip install scipy")

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
    print("✓ Matplotlib loaded successfully")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: Matplotlib not available. Install with: pip install matplotlib")

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
    print("✓ Vosk loaded successfully")
except ImportError:
    VOSK_AVAILABLE = False
    print("WARNING: Vosk not available. Install with: pip install vosk")

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
    print("✓ SpeechRecognition loaded successfully")
except ImportError:
    SR_AVAILABLE = False
    print("WARNING: speech_recognition not available")

try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✓ OpenAI Whisper loaded successfully")
except ImportError:
    WHISPER_AVAILABLE = False
    print("WARNING: OpenAI Whisper not available. Install with: pip install openai-whisper")


# Available Vosk models
VOSK_MODELS = {
    'vosk-model-small-en-us-0.15': {
        'name': 'vosk-model-small-en-us-0.15',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
        'size': '40 MB',
        'description': 'Small English model - Fast, good for testing',
        'accuracy': '⭐⭐⭐'
    },
    'vosk-model-en-us-0.22-lgraph': {
        'name': 'vosk-model-en-us-0.22-lgraph',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip',
        'size': '128 MB',
        'description': 'Medium English model - Balanced speed and accuracy',
        'accuracy': '⭐⭐⭐⭐'
    },
    'vosk-model-en-us-0.22': {
        'name': 'vosk-model-en-us-0.22',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip',
        'size': '1.8 GB',
        'description': 'Large English model - Best accuracy',
        'accuracy': '⭐⭐⭐⭐⭐'
    }
}


class BandpassFilter:
    """Real-time bandpass filter implementation"""
    def __init__(self, lowcut=300, highcut=3400, fs=48000, order=5):
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for filtering")
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.sos = self._design_filter()
        self.zi = signal.sosfilt_zi(self.sos)
    
    def _design_filter(self):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        sos = signal.butter(self.order, [low, high], btype='band', output='sos')
        return sos
    
    def filter(self, data):
        """Apply filter to audio data"""
        filtered, self.zi = signal.sosfilt(self.sos, data, zi=self.zi)
        return filtered.astype(np.int16)
    
    def update_params(self, lowcut, highcut):
        """Update filter parameters"""
        self.lowcut = lowcut
        self.highcut = highcut
        self.sos = self._design_filter()
        self.zi = signal.sosfilt_zi(self.sos)


class AudioRecorder(QThread):
    """Background thread for audio recording"""
    audio_data_ready = pyqtSignal(np.ndarray)
    recording_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, device_index, sample_rate=48000, channels=1):
        super().__init__()
        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio is required for recording")
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.is_recording = False
        self.audio_frames = []
        self.filter_enabled = False
        self.bandpass_filter = None
        self.filename = ""
    
    def set_filter(self, enabled, lowcut=300, highcut=3400):
        self.filter_enabled = enabled
        if enabled and SCIPY_AVAILABLE:
            self.bandpass_filter = BandpassFilter(lowcut, highcut, self.sample_rate)
    
    def run(self):
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk
            )
            
            self.is_recording = True
            self.audio_frames = []
            
            while self.is_recording:
                data = stream.read(self.chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Apply filter if enabled
                if self.filter_enabled and self.bandpass_filter:
                    audio_data = self.bandpass_filter.filter(audio_data)
                
                self.audio_frames.append(audio_data.tobytes())
                self.audio_data_ready.emit(audio_data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save recording
            if self.filename and self.audio_frames:
                wf = wave.open(self.filename, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_frames))
                wf.close()
                self.recording_finished.emit(self.filename)
        
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def stop(self):
        self.is_recording = False


class LiveAudioProcessor(QThread):
    """Background thread for live audio monitoring with volume control"""
    audio_level = pyqtSignal(float)
    
    def __init__(self, device_index, sample_rate=48000):
        super().__init__()
        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio is required for live processing")
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk = 1024
        self.is_running = False
        self.filter_enabled = False
        self.bandpass_filter = None
        self.volume_multiplier = 1.0
    
    def set_filter(self, enabled, lowcut=300, highcut=3400):
        self.filter_enabled = enabled
        if enabled and SCIPY_AVAILABLE:
            self.bandpass_filter = BandpassFilter(lowcut, highcut, self.sample_rate)
    
    def set_volume(self, multiplier):
        """Set volume amplification (1-10x)"""
        self.volume_multiplier = max(1.0, min(10.0, multiplier))
    
    def run(self):
        try:
            p = pyaudio.PyAudio()
            
            input_stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk
            )
            
            output_stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk
            )
            
            self.is_running = True
            
            while self.is_running:
                data = input_stream.read(self.chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                
                # Apply filter if enabled
                if self.filter_enabled and self.bandpass_filter:
                    audio_data = self.bandpass_filter.filter(audio_data.astype(np.int16)).astype(np.float32)
                
                # Apply volume amplification
                audio_data = audio_data * self.volume_multiplier
                
                # Clip to prevent distortion
                audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)
                
                # Calculate volume level
                level = np.abs(audio_data).mean() / 32768.0 * 100
                self.audio_level.emit(level)
                
                # Play filtered audio
                output_stream.write(audio_data.tobytes())
            
            input_stream.stop_stream()
            input_stream.close()
            output_stream.stop_stream()
            output_stream.close()
            p.terminate()
        
        except Exception as e:
            print(f"Live audio error: {e}")
    
    def stop(self):
        self.is_running = False


class TranscriptionThread(QThread):
    """Background thread for audio transcription"""
    transcription_ready = pyqtSignal(str, str)  # (method, text)
    progress_update = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, audio_file, methods=['vosk'], vosk_model='vosk-model-small-en-us-0.15', whisper_model='small', ffmpeg_path=None):
        super().__init__()
        self.audio_file = audio_file
        self.methods = methods
        self.vosk_model = vosk_model
        self.whisper_model = whisper_model
        self.ffmpeg_path = ffmpeg_path
        self.converted_file = None
    
    def run(self):
        try:
            # Convert audio to 16kHz mono WAV for compatibility
            self.converted_file = self.convert_audio_to_16khz(self.audio_file)
            
            for method in self.methods:
                if method == 'vosk' and VOSK_AVAILABLE:
                    self.transcribe_vosk()
                elif method == 'sr' and SR_AVAILABLE:
                    self.transcribe_speech_recognition()
                elif method == 'whisper' and WHISPER_AVAILABLE:
                    self.transcribe_whisper()
            
            # Clean up converted file
            if self.converted_file and self.converted_file != self.audio_file and os.path.exists(self.converted_file):
                try:
                    os.remove(self.converted_file)
                except:
                    pass
                    
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def convert_audio_to_16khz(self, input_file):
        """Convert audio to 16kHz mono WAV for transcription compatibility"""
        try:
            temp_dir = os.path.dirname(input_file)
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            temp_file = os.path.join(temp_dir, f"temp_16khz_{base_name}.wav")
            
            # Check if ffmpeg is available (use provided path if given)
            ffmpeg_available = is_ffmpeg_available(self.ffmpeg_path)

            if ffmpeg_available:
                # Resolve ffmpeg executable to call
                if self.ffmpeg_path:
                    if os.path.isdir(self.ffmpeg_path):
                        ffmpeg_exec = os.path.join(self.ffmpeg_path, 'ffmpeg.exe')
                    else:
                        ffmpeg_exec = self.ffmpeg_path
                else:
                    ffmpeg_exec = 'ffmpeg'

                cmd = [
                    ffmpeg_exec, '-i', input_file,
                    '-ar', '16000',
                    '-ac', '1',
                    '-f', 'wav',
                    '-y',
                    temp_file
                ]

                subprocess.run(cmd, capture_output=True, check=True, timeout=60)
                print(f"✓ Converted audio to 16kHz using ffmpeg: {temp_file}")
                return temp_file
            else:
                print("⚠ ffmpeg not found, using scipy for conversion...")
                
                if not SCIPY_AVAILABLE:
                    print("⚠ scipy not available, using original file")
                    return input_file

                sample_rate, data = wavfile.read(input_file)

                if len(data.shape) > 1:
                    data = data.mean(axis=1).astype(data.dtype)

                if sample_rate != 16000:
                    from scipy import signal as sp_signal
                    num_samples = int(len(data) * 16000 / sample_rate)
                    data = sp_signal.resample(data, num_samples)
                    data = data.astype(np.int16)

                wavfile.write(temp_file, 16000, data)
                print(f"✓ Converted audio to 16kHz using scipy: {temp_file}")
                return temp_file
                
        except Exception as e:
            print(f"Audio conversion error: {e}, using original file")
            return input_file
    
    def transcribe_vosk(self):
        """Transcribe using Vosk offline"""
        model_path = os.path.join("models", self.vosk_model)
        
        if not os.path.exists(model_path):
            error_msg = f"Vosk model not found at: {model_path}\n\n"
            error_msg += f"Please download from:\n{VOSK_MODELS.get(self.vosk_model, {}).get('url', 'https://alphacephei.com/vosk/models')}\n\n"
            error_msg += "Extract the model to the 'models' folder."
            self.error_occurred.emit(error_msg)
            return
        
        try:
            model = Model(model_path)
            audio_file = self.converted_file if self.converted_file else self.audio_file
            wf = wave.open(audio_file, "rb")
            
            if wf.getframerate() != 16000:
                print(f"⚠ Warning: Vosk expects 16kHz audio, got {wf.getframerate()}Hz")
            
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)
            
            results = []
            total_frames = wf.getnframes()
            frames_read = 0
            
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                
                frames_read += 4000
                progress = int((frames_read / total_frames) * 33)
                self.progress_update.emit(progress)
                
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if 'text' in result and result['text']:
                        results.append(result['text'])
            
            final_result = json.loads(rec.FinalResult())
            if 'text' in final_result and final_result['text']:
                results.append(final_result['text'])
            
            wf.close()
            
            transcription = ' '.join(results)
            if not transcription.strip():
                transcription = "[No speech detected by Vosk]"
            
            self.transcription_ready.emit('vosk', transcription)
        
        except Exception as e:
            self.error_occurred.emit(f"Vosk error: {str(e)}")
    
    def transcribe_speech_recognition(self):
        """Transcribe using Python Speech Recognition"""
        try:
            recognizer = sr.Recognizer()
            audio_file = self.converted_file if self.converted_file else self.audio_file
            # Validate that audio_file is a readable WAV with suitable parameters
            try:
                # Try open with wave to detect common format issues
                with wave.open(audio_file, 'rb') as wf:
                    nch = wf.getnchannels()
                    sampw = wf.getsampwidth()
                    fr = wf.getframerate()
            except Exception as e:
                self.error_occurred.emit(f"SpeechRecognition: cannot open audio file: {e}\nEnsure ffmpeg or SciPy is installed to convert files to 16kHz mono WAV.")
                return

            # If not mono 16-bit, try to convert (scipy fallback already attempted in convert_audio_to_16khz)
            if nch != 1 or sampw != 2 or fr not in [8000, 16000, 32000, 48000]:
                # Still try to proceed since sr.AudioFile can handle some rates, but warn
                print(f"SpeechRecognition: audio format is nch={nch}, sampw={sampw}, fr={fr}. Proceeding but results may vary.")

            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)

            self.progress_update.emit(66)

            try:
                transcription = recognizer.recognize_google(audio)
                self.progress_update.emit(100)
                self.transcription_ready.emit('sr', transcription)
            except sr.UnknownValueError:
                self.transcription_ready.emit('sr', "[Could not understand audio]")
                self.progress_update.emit(100)
            except sr.RequestError as e:
                self.error_occurred.emit(f"Speech Recognition request error: {e}\nEnsure you have network connectivity for Google SR.")
        
        except Exception as e:
            self.error_occurred.emit(f"Speech Recognition error: {str(e)}")
    
    def transcribe_whisper(self):
        """Transcribe using OpenAI Whisper"""
        try:
            self.progress_update.emit(70)
            
            print("Loading Whisper model (this may take a moment)...")
            # Use the selected whisper model name passed to the thread
            model_name = getattr(self, 'whisper_model', 'small')

            # Check ffmpeg availability because whisper uses ffmpeg for many formats
            if not is_ffmpeg_available(self.ffmpeg_path):
                # If the converted file exists and is a WAV, whisper can still read it; otherwise fail fast
                audio_candidate = self.converted_file if self.converted_file else self.audio_file
                try:
                    with wave.open(audio_candidate, 'rb') as wf:
                        pass
                except Exception:
                    self.error_occurred.emit("Whisper requires ffmpeg or a valid converted WAV. Provide an ffmpeg path in the Transcription Settings or install ffmpeg on PATH.")
                    return

            model = None
            try:
                # Ensure ffmpeg path (if provided) is on PATH temporarily for whisper internals
                if self.ffmpeg_path and os.path.exists(self.ffmpeg_path):
                    old_path = os.environ.get('PATH', '')
                    if os.path.isdir(self.ffmpeg_path):
                        ff_dir = self.ffmpeg_path
                    else:
                        ff_dir = os.path.dirname(self.ffmpeg_path)
                    os.environ['PATH'] = ff_dir + os.pathsep + old_path
                    try:
                        model = whisper.load_model(model_name)
                    finally:
                        os.environ['PATH'] = old_path
                else:
                    model = whisper.load_model(model_name)
            except FileNotFoundError as e:
                # Likely missing model files or download failure
                self.error_occurred.emit(f"Whisper model load error: {e}. Check model name and internet access to download models.")
                return
            except Exception as e:
                self.error_occurred.emit(f"Whisper load_model error: {e}")
                return
            
            self.progress_update.emit(80)
            
            audio_file = self.converted_file if self.converted_file else self.audio_file
            # Ensure the audio file exists and is readable
            if not os.path.exists(audio_file):
                self.error_occurred.emit(f"Whisper error: converted audio file not found: {audio_file}")
                return

            try:
                if self.ffmpeg_path and os.path.exists(self.ffmpeg_path):
                    old_path = os.environ.get('PATH', '')
                    if os.path.isdir(self.ffmpeg_path):
                        ff_dir = self.ffmpeg_path
                    else:
                        ff_dir = os.path.dirname(self.ffmpeg_path)
                    os.environ['PATH'] = ff_dir + os.pathsep + old_path
                    try:
                        result = model.transcribe(audio_file, fp16=False)
                    finally:
                        os.environ['PATH'] = old_path
                else:
                    result = model.transcribe(audio_file, fp16=False)
            except FileNotFoundError as e:
                # This commonly indicates ffmpeg is missing on Windows (WinError 2)
                self.error_occurred.emit("Whisper error: ffmpeg not found on PATH. Install ffmpeg and restart the app. See https://ffmpeg.org/download.html")
                return
            except Exception as e:
                self.error_occurred.emit(f"Whisper error: {str(e)}")
                return
            
            transcription = result['text'].strip()
            if not transcription:
                transcription = "[No speech detected by Whisper]"
            
            self.progress_update.emit(100)
            self.transcription_ready.emit('whisper', transcription)
        
        except Exception as e:
            self.error_occurred.emit(f"Whisper error: {str(e)}")


if MATPLOTLIB_AVAILABLE:
    class MplCanvas(FigureCanvasQTAgg):
        """Matplotlib canvas for Qt"""
        def __init__(self, parent=None, width=5, height=4, dpi=100):
            fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1a1a2e')
            self.axes = fig.add_subplot(111)
            self.axes.set_facecolor('#0f0f1e')
            super().__init__(fig)


class LaserMicrophoneApp(QMainWindow):
    def __init__(self):
        super().__init__()
        print("Initializing Laser Microphone App...")
        
        self.setWindowTitle("Laser Microphone Suite - Advanced Audio Processing")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(self.get_stylesheet())
        
        self.audio_devices = []
        self.selected_device = 0
        
        if PYAUDIO_AVAILABLE:
            try:
                self.p = pyaudio.PyAudio()
                self.audio_devices = self.get_audio_devices()
                print(f"✓ Found {len(self.audio_devices)} audio devices")
            except Exception as e:
                print(f"WARNING: PyAudio initialization failed: {e}")
                self.p = None
        else:
            self.p = None
            print("WARNING: PyAudio not available - recording features disabled")
        
        self.recorder = None
        self.live_processor = None
        self.is_recording = False
        self.is_live_listening = False
        
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.positionChanged.connect(self.on_playback_position_changed)
        self.media_player.durationChanged.connect(self.on_playback_duration_changed)
        self.is_playing = False
        
        self.recordings_dir = "recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
        self.recordings = self.load_recordings()
        
        self.filter_enabled = True
        self.lowcut = 300
        self.highcut = 3400
        self.volume_multiplier = 1.0
        
        self.selected_vosk_model = 'vosk-model-small-en-us-0.15'
        
        try:
            self.init_ui()
            print("✓ UI initialized successfully")
        except Exception as e:
            print(f"ERROR initializing UI: {e}")
            traceback.print_exc()
        
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_time)
        self.recording_duration = 0
        
        print("✓ Application initialized successfully")
    
    def get_audio_devices(self):
        """Get list of available audio input devices"""
        devices = []
        if not self.p:
            return devices
        
        try:
            for i in range(self.p.get_device_count()):
                info = self.p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': int(info['defaultSampleRate'])
                    })
        except Exception as e:
            print(f"Error getting audio devices: {e}")
        
        return devices
    
    def load_recordings(self):
        """Load recordings from database file"""
        db_file = os.path.join(self.recordings_dir, "recordings.json")
        if os.path.exists(db_file):
            try:
                with open(db_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_recordings(self):
        """Save recordings to database file"""
        db_file = os.path.join(self.recordings_dir, "recordings.json")
        with open(db_file, 'w') as f:
            json.dump(self.recordings, f, indent=2)
    
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 3)
    
    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        header = QLabel("🎤 Laser Microphone")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header.setStyleSheet("color: #a855f7; padding: 10px;")
        layout.addWidget(header)
        
        device_group = QGroupBox("Audio Input Device")
        device_layout = QVBoxLayout()
        self.device_combo = QComboBox()
        
        if self.audio_devices:
            for device in self.audio_devices:
                self.device_combo.addItem(f"{device['name']} ({device['sample_rate']} Hz)", device['index'])
            self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        else:
            self.device_combo.addItem("No audio devices found", -1)
            self.device_combo.setEnabled(False)
        
        device_layout.addWidget(self.device_combo)
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        recording_group = QGroupBox("Recording")
        recording_layout = QVBoxLayout()
        
        name_label = QLabel("Recording Name:")
        recording_layout.addWidget(name_label)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter recording name...")
        recording_layout.addWidget(self.name_input)
        
        filter_layout = QHBoxLayout()
        self.filter_checkbox = QCheckBox("Bandpass Filter")
        self.filter_checkbox.setChecked(True)
        self.filter_checkbox.stateChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.filter_checkbox)
        recording_layout.addLayout(filter_layout)
        
        freq_layout = QVBoxLayout()
        
        lowcut_layout = QHBoxLayout()
        lowcut_label = QLabel(f"Low Cut: {self.lowcut} Hz")
        self.lowcut_label = lowcut_label
        lowcut_layout.addWidget(lowcut_label)
        self.lowcut_spin = QSpinBox()
        self.lowcut_spin.setRange(20, 8000)
        self.lowcut_spin.setValue(self.lowcut)
        self.lowcut_spin.valueChanged.connect(self.on_lowcut_changed)
        lowcut_layout.addWidget(self.lowcut_spin)
        freq_layout.addLayout(lowcut_layout)
        
        highcut_layout = QHBoxLayout()
        highcut_label = QLabel(f"High Cut: {self.highcut} Hz")
        self.highcut_label = highcut_label
        highcut_layout.addWidget(highcut_label)
        self.highcut_spin = QSpinBox()
        self.highcut_spin.setRange(100, 20000)
        self.highcut_spin.setValue(self.highcut)
        self.highcut_spin.valueChanged.connect(self.on_highcut_changed)
        highcut_layout.addWidget(self.highcut_spin)
        freq_layout.addLayout(highcut_layout)
        
        recording_layout.addLayout(freq_layout)
        
        volume_layout = QVBoxLayout()
        volume_label = QLabel(f"Volume Amplification: {self.volume_multiplier:.1f}x")
        self.volume_amp_label = volume_label
        volume_layout.addWidget(volume_label)
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(10)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(10)
        self.volume_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.volume_slider.setTickInterval(10)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        volume_layout.addWidget(self.volume_slider)
        recording_layout.addLayout(volume_layout)
        
        self.volume_label = QLabel("Input Level: 0%")
        recording_layout.addWidget(self.volume_label)
        self.volume_bar = QProgressBar()
        self.volume_bar.setMaximum(100)
        recording_layout.addWidget(self.volume_bar)
        
        self.time_label = QLabel("00:00")
        self.time_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("color: #a855f7;")
        recording_layout.addWidget(self.time_label)
        
        self.record_btn = QPushButton("🔴 Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setMinimumHeight(50)
        if not PYAUDIO_AVAILABLE:
            self.record_btn.setEnabled(False)
            self.record_btn.setText("PyAudio Required")
        recording_layout.addWidget(self.record_btn)
        
        self.live_btn = QPushButton("🔊 Start Live Listening")
        self.live_btn.clicked.connect(self.toggle_live_listening)
        self.live_btn.setMinimumHeight(40)
        if not PYAUDIO_AVAILABLE:
            self.live_btn.setEnabled(False)
            self.live_btn.setText("PyAudio Required")
        recording_layout.addWidget(self.live_btn)
        
        recording_group.setLayout(recording_layout)
        layout.addWidget(recording_group)
        
        stats_group = QGroupBox("Session Stats")
        stats_layout = QVBoxLayout()
        
        self.stats_recordings = QLabel(f"Total Recordings: {len(self.recordings)}")
        stats_layout.addWidget(self.stats_recordings)
        
        stats_layout.addWidget(QLabel("Sample Rate: 48 kHz"))
        stats_layout.addWidget(QLabel("Bit Depth: 16-bit"))
        stats_layout.addWidget(QLabel("Format: WAV Mono"))
        
        status_text = "\nLibrary Status:\n"
        status_text += f"• PyAudio: {'✓' if PYAUDIO_AVAILABLE else '✗'}\n"
        status_text += f"• SciPy: {'✓' if SCIPY_AVAILABLE else '✗'}\n"
        status_text += f"• Matplotlib: {'✓' if MATPLOTLIB_AVAILABLE else '✗'}\n"
        status_text += f"• Vosk: {'✓' if VOSK_AVAILABLE else '✗'}\n"
        status_text += f"• SpeechRecognition: {'✓' if SR_AVAILABLE else '✗'}\n"
        status_text += f"• Whisper: {'✓' if WHISPER_AVAILABLE else '✗'}\n"
        # FFmpeg availability on PATH (initial check)
        ff_on_path = is_ffmpeg_available(None)
        status_text += f"• FFmpeg on PATH: {'✓' if ff_on_path else '✗'}"
        self.ffmpeg_status_label = QLabel(status_text)
        stats_layout.addWidget(self.ffmpeg_status_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """Create right content panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        self.tabs = QTabWidget()
        
        recordings_tab = self.create_recordings_tab()
        self.tabs.addTab(recordings_tab, "📁 Recordings")
        
        visualize_tab = self.create_visualize_tab()
        self.tabs.addTab(visualize_tab, "📊 Visualize")
        
        transcribe_tab = self.create_transcribe_tab()
        self.tabs.addTab(transcribe_tab, "📝 Transcribe")
        
        layout.addWidget(self.tabs)
        
        return panel
    
    def create_recordings_tab(self):
        """Create recordings list tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.recordings_list = QListWidget()
        self.recordings_list.itemClicked.connect(self.on_recording_selected)
        self.update_recordings_list()
        layout.addWidget(self.recordings_list)
        
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QVBoxLayout()
        
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        self.current_time_label.setStyleSheet("color: #a855f7; font-weight: bold;")
        time_layout.addWidget(self.current_time_label)
        
        time_layout.addStretch()
        
        self.total_time_label = QLabel("00:00")
        self.total_time_label.setStyleSheet("color: #888;")
        time_layout.addWidget(self.total_time_label)
        playback_layout.addLayout(time_layout)
        
        self.playback_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_slider.setMinimum(0)
        self.playback_slider.setMaximum(1000)
        self.playback_slider.setValue(0)
        self.playback_slider.sliderMoved.connect(self.on_playback_slider_moved)
        self.playback_slider.setEnabled(False)
        playback_layout.addWidget(self.playback_slider)
        
        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)
        
        btn_layout = QHBoxLayout()
        
        upload_btn = QPushButton("📤 Upload")
        upload_btn.clicked.connect(self.upload_recording)
        btn_layout.addWidget(upload_btn)
        
        play_btn = QPushButton("▶ Play")
        play_btn.clicked.connect(self.play_recording)
        btn_layout.addWidget(play_btn)
        
        pause_btn = QPushButton("⏸ Pause")
        pause_btn.clicked.connect(self.pause_playback)
        btn_layout.addWidget(pause_btn)
        
        stop_btn = QPushButton("⏹ Stop")
        stop_btn.clicked.connect(self.stop_playback)
        btn_layout.addWidget(stop_btn)
        
        transcribe_btn = QPushButton("📝 Transcribe")
        transcribe_btn.clicked.connect(self.transcribe_selected)
        btn_layout.addWidget(transcribe_btn)
        
        export_btn = QPushButton("💾 Export")
        export_btn.clicked.connect(self.export_recording)
        btn_layout.addWidget(export_btn)
        
        delete_btn = QPushButton("🗑 Delete")
        delete_btn.clicked.connect(self.delete_recording)
        btn_layout.addWidget(delete_btn)
        
        layout.addLayout(btn_layout)
        
        return tab
    
    def create_visualize_tab(self):
        """Create visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        if MATPLOTLIB_AVAILABLE and SCIPY_AVAILABLE:
            waveform_label = QLabel("Waveform")
            waveform_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            layout.addWidget(waveform_label)
            
            self.waveform_canvas = MplCanvas(self, width=8, height=3, dpi=100)
            layout.addWidget(self.waveform_canvas)
            
            spectrogram_label = QLabel("Spectrogram")
            spectrogram_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            layout.addWidget(spectrogram_label)
            
            self.spectrogram_canvas = MplCanvas(self, width=8, height=3, dpi=100)
            layout.addWidget(self.spectrogram_canvas)
            
            load_btn = QPushButton("Load Selected Recording for Visualization")
            load_btn.clicked.connect(self.visualize_selected)
            layout.addWidget(load_btn)
        else:
            missing = []
            if not MATPLOTLIB_AVAILABLE:
                missing.append("Matplotlib")
            if not SCIPY_AVAILABLE:
                missing.append("SciPy")
            layout.addWidget(QLabel(f"{' and '.join(missing)} required for visualization.\nInstall with: pip install {' '.join(missing).lower()}"))
        
        return tab
    
    def create_transcribe_tab(self):
        """Create transcription tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        model_group = QGroupBox("Transcription Settings")
        model_layout = QVBoxLayout()
        
        if VOSK_AVAILABLE:
            vosk_label = QLabel("Vosk Model:")
            model_layout.addWidget(vosk_label)
            
            self.vosk_model_combo = QComboBox()
            for model_key, model_info in VOSK_MODELS.items():
                display_text = f"{model_info['name']} - {model_info['size']} {model_info['accuracy']}"
                self.vosk_model_combo.addItem(display_text, model_key)
            self.vosk_model_combo.currentIndexChanged.connect(self.on_vosk_model_changed)
            model_layout.addWidget(self.vosk_model_combo)
            
            self.model_info_label = QLabel()
            self.update_model_info()
            model_layout.addWidget(self.model_info_label)
        
        method_label = QLabel("Transcription Methods:")
        model_layout.addWidget(method_label)
        
        self.vosk_checkbox = QCheckBox("Vosk (Offline)")
        self.vosk_checkbox.setChecked(VOSK_AVAILABLE)
        self.vosk_checkbox.setEnabled(VOSK_AVAILABLE)
        model_layout.addWidget(self.vosk_checkbox)
        
        self.sr_checkbox = QCheckBox("Google Speech Recognition (Online)")
        self.sr_checkbox.setChecked(SR_AVAILABLE)
        self.sr_checkbox.setEnabled(SR_AVAILABLE)
        model_layout.addWidget(self.sr_checkbox)
        
        self.whisper_checkbox = QCheckBox("OpenAI Whisper small (Offline - Best Quality)")
        self.whisper_checkbox.setChecked(False)
        self.whisper_checkbox.setEnabled(WHISPER_AVAILABLE)
        model_layout.addWidget(self.whisper_checkbox)
        
        # Whisper model selector (small / medium)
        if WHISPER_AVAILABLE:
            whisper_model_layout = QHBoxLayout()
            whisper_model_label = QLabel("Whisper Model:")
            whisper_model_layout.addWidget(whisper_model_label)
            self.whisper_model_combo = QComboBox()
            self.whisper_model_combo.addItem("small", "small")
            self.whisper_model_combo.addItem("medium", "medium")
            self.whisper_model_combo.setCurrentIndex(0)
            whisper_model_layout.addWidget(self.whisper_model_combo)
            model_layout.addLayout(whisper_model_layout)

        if not WHISPER_AVAILABLE:
            whisper_info = QLabel("Install Whisper: pip install openai-whisper")
            whisper_info.setStyleSheet("color: #888; font-size: 9px;")
            model_layout.addWidget(whisper_info)
        
        # FFmpeg path entry (optional) - allows using a local ffmpeg binary that's not on PATH
        ffmpeg_layout = QHBoxLayout()
        ffmpeg_label = QLabel("FFmpeg path (optional):")
        ffmpeg_layout.addWidget(ffmpeg_label)
        self.ffmpeg_input = QLineEdit()
        self.ffmpeg_input.setPlaceholderText("C:\\path\\to\\ffmpeg.exe or folder containing ffmpeg.exe")
        ffmpeg_layout.addWidget(self.ffmpeg_input)
        ffmpeg_check_btn = QPushButton("Check")
        def _check_ffmpeg():
            p = self.ffmpeg_input.text().strip()
            ok = is_ffmpeg_available(p if p else None)
            QMessageBox.information(self, "FFmpeg Check", f"FFmpeg available: {ok}\nPath checked: {p if p else 'system PATH'}")
            # Update status label in Session Stats if present
            try:
                ff_on_path = is_ffmpeg_available(None)
                if hasattr(self, 'ffmpeg_status_label'):
                    prev = self.ffmpeg_status_label.text().split('\n')
                    # replace last line with FFmpeg status
                    if prev:
                        prev[-1] = f"• FFmpeg on PATH: {'✓' if ff_on_path else '✗'}"
                        self.ffmpeg_status_label.setText('\n'.join(prev))
            except Exception:
                pass
        ffmpeg_check_btn.clicked.connect(_check_ffmpeg)
        ffmpeg_layout.addWidget(ffmpeg_check_btn)
        model_layout.addLayout(ffmpeg_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        self.transcribe_progress = QProgressBar()
        layout.addWidget(self.transcribe_progress)
        
        transcribe_label = QLabel("Transcriptions:")
        transcribe_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(transcribe_label)
        
        transcribe_container = QWidget()
        transcribe_layout = QHBoxLayout(transcribe_container)
        
        vosk_group = QGroupBox("Vosk Transcription")
        vosk_layout = QVBoxLayout()
        self.vosk_text = QTextEdit()
        self.vosk_text.setReadOnly(True)
        self.vosk_text.setPlaceholderText("Vosk transcription will appear here...")
        vosk_layout.addWidget(self.vosk_text)
        vosk_group.setLayout(vosk_layout)
        transcribe_layout.addWidget(vosk_group)
        
        sr_group = QGroupBox("Google Speech Recognition")
        sr_layout = QVBoxLayout()
        self.sr_text = QTextEdit()
        self.sr_text.setReadOnly(True)
        self.sr_text.setPlaceholderText("Speech Recognition transcription will appear here...")
        sr_layout.addWidget(self.sr_text)
        sr_group.setLayout(sr_layout)
        transcribe_layout.addWidget(sr_group)
        
        whisper_group = QGroupBox("OpenAI Whisper")
        whisper_layout = QVBoxLayout()
        self.whisper_text = QTextEdit()
        self.whisper_text.setReadOnly(True)
        self.whisper_text.setPlaceholderText("Whisper transcription will appear here...")
        whisper_layout.addWidget(self.whisper_text)
        whisper_group.setLayout(whisper_layout)
        transcribe_layout.addWidget(whisper_group)
        
        layout.addWidget(transcribe_container)
        
        btn_layout = QHBoxLayout()
        
        start_transcribe_btn = QPushButton("🎙 Start Transcription")
        start_transcribe_btn.clicked.connect(self.start_transcription)
        btn_layout.addWidget(start_transcribe_btn)
        
        export_vosk_btn = QPushButton("💾 Export Vosk")
        export_vosk_btn.clicked.connect(lambda: self.export_transcription('vosk'))
        btn_layout.addWidget(export_vosk_btn)
        
        export_sr_btn = QPushButton("💾 Export Google SR")
        export_sr_btn.clicked.connect(lambda: self.export_transcription('sr'))
        btn_layout.addWidget(export_sr_btn)
        
        export_whisper_btn = QPushButton("💾 Export Whisper")
        export_whisper_btn.clicked.connect(lambda: self.export_transcription('whisper'))
        btn_layout.addWidget(export_whisper_btn)
        
        export_all_btn = QPushButton("💾 Export All")
        export_all_btn.clicked.connect(lambda: self.export_transcription('all'))
        btn_layout.addWidget(export_all_btn)
        
        layout.addLayout(btn_layout)
        
        return tab
    
    def on_device_changed(self, index):
        """Handle device selection change"""
        self.selected_device = self.device_combo.itemData(index)
    
    def on_filter_changed(self, state):
        """Handle filter checkbox change"""
        self.filter_enabled = state == Qt.CheckState.Checked.value
    
    def on_lowcut_changed(self, value):
        """Handle low cut frequency change"""
        self.lowcut = value
        self.lowcut_label.setText(f"Low Cut: {value} Hz")
    
    def on_highcut_changed(self, value):
        """Handle high cut frequency change"""
        self.highcut = value
        self.highcut_label.setText(f"High Cut: {value} Hz")
    
    def on_volume_changed(self, value):
        """Handle volume slider change"""
        self.volume_multiplier = value / 10.0
        self.volume_amp_label.setText(f"Volume Amplification: {self.volume_multiplier:.1f}x")
        if self.live_processor and self.is_live_listening:
            self.live_processor.set_volume(self.volume_multiplier)
    
    def on_vosk_model_changed(self, index):
        """Handle Vosk model selection change"""
        self.selected_vosk_model = self.vosk_model_combo.itemData(index)
        self.update_model_info()
    
    def update_model_info(self):
        """Update model information display"""
        if VOSK_AVAILABLE and hasattr(self, 'model_info_label'):
            model_info = VOSK_MODELS.get(self.selected_vosk_model, {})
            info_text = f"Description: {model_info.get('description', 'N/A')}\n"
            info_text += f"Download: {model_info.get('url', 'N/A')}"
            self.model_info_label.setText(info_text)
            self.model_info_label.setWordWrap(True)
            self.model_info_label.setStyleSheet("color: #888; font-size: 10px;")
    
    def toggle_recording(self):
        """Start or stop recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start audio recording"""
        if not PYAUDIO_AVAILABLE:
            QMessageBox.warning(self, "Error", "PyAudio is not available")
            return
        
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a recording name")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.recordings_dir, f"{name}_{timestamp}.wav")
            
            self.recorder = AudioRecorder(self.selected_device)
            self.recorder.set_filter(self.filter_enabled, self.lowcut, self.highcut)
            self.recorder.filename = filename
            self.recorder.audio_data_ready.connect(self.on_audio_data)
            self.recorder.recording_finished.connect(self.on_recording_finished)
            self.recorder.error_occurred.connect(self.on_error)
            self.recorder.start()
            
            self.is_recording = True
            self.record_btn.setText("⏹ Stop Recording")
            self.record_btn.setStyleSheet("background-color: #dc2626;")
            self.name_input.setEnabled(False)
            
            self.recording_duration = 0
            self.recording_timer.start(1000)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start recording: {str(e)}")
    
    def stop_recording(self):
        """Stop audio recording"""
        if self.recorder:
            self.recorder.stop()
        
        self.is_recording = False
        self.record_btn.setText("🔴 Start Recording")
        self.record_btn.setStyleSheet("")
        self.name_input.setEnabled(True)
        self.recording_timer.stop()
        self.time_label.setText("00:00")
        self.volume_bar.setValue(0)
    
    def toggle_live_listening(self):
        """Start or stop live listening"""
        if not self.is_live_listening:
            self.start_live_listening()
        else:
            self.stop_live_listening()
    
    def start_live_listening(self):
        """Start live audio monitoring"""
        if not PYAUDIO_AVAILABLE:
            QMessageBox.warning(self, "Error", "PyAudio is not available")
            return
        
        try:
            self.live_processor = LiveAudioProcessor(self.selected_device)
            self.live_processor.set_filter(self.filter_enabled, self.lowcut, self.highcut)
            self.live_processor.set_volume(self.volume_multiplier)
            self.live_processor.audio_level.connect(self.update_volume_level)
            self.live_processor.start()
            
            self.is_live_listening = True
            self.live_btn.setText("⏹ Stop Live Listening")
            self.live_btn.setStyleSheet("background-color: #16a34a;")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start live listening: {str(e)}")
    
    def stop_live_listening(self):
        """Stop live audio monitoring"""
        if self.live_processor:
            self.live_processor.stop()
            self.live_processor.wait()
        
        self.is_live_listening = False
        self.live_btn.setText("🔊 Start Live Listening")
        self.live_btn.setStyleSheet("")
        self.volume_bar.setValue(0)
    
    def on_audio_data(self, data):
        """Handle incoming audio data"""
        level = np.abs(data).mean() / 32768.0 * 100
        self.update_volume_level(level)
    
    def update_volume_level(self, level):
        """Update volume meter"""
        self.volume_bar.setValue(int(level))
        self.volume_label.setText(f"Input Level: {int(level)}%")
    
    def update_recording_time(self):
        """Update recording timer"""
        self.recording_duration += 1
        mins = self.recording_duration // 60
        secs = self.recording_duration % 60
        self.time_label.setText(f"{mins:02d}:{secs:02d}")
    
    def on_recording_finished(self, filename):
        """Handle recording completion"""
        recording_data = {
            'name': self.name_input.text(),
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'duration': self.recording_duration,
            'filtered': self.filter_enabled,
            'lowcut': self.lowcut if self.filter_enabled else None,
            'highcut': self.highcut if self.filter_enabled else None,
            'vosk_transcription': '',
            'sr_transcription': '',
            'whisper_transcription': ''
        }
        
        self.recordings.append(recording_data)
        self.save_recordings()
        self.update_recordings_list()
        self.stats_recordings.setText(f"Total Recordings: {len(self.recordings)}")
        
        QMessageBox.information(self, "Success", f"Recording saved: {filename}")
        self.name_input.clear()
    
    def on_error(self, error_msg):
        """Handle errors"""
        QMessageBox.critical(self, "Error", error_msg)
    
    def update_recordings_list(self):
        """Update recordings list widget"""
        self.recordings_list.clear()
        for rec in reversed(self.recordings):
            item_text = f"{rec['name']} - {rec['timestamp'][:10]} ({rec['duration']}s)"
            if rec['filtered']:
                item_text += f" [Filtered: {rec['lowcut']}-{rec['highcut']} Hz]"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, rec)
            self.recordings_list.addItem(item)
    
    def on_recording_selected(self, item):
        """Handle recording selection"""
        self.selected_recording = item.data(Qt.ItemDataRole.UserRole)
    
    def upload_recording(self):
        """Upload an external recording"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Upload Recording", 
            "",
            "WAV Files (*.wav);;All Files (*.*)"
        )
        
        if filename:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(os.path.basename(filename))[0]
                new_filename = os.path.join(self.recordings_dir, f"{base_name}_{timestamp}.wav")
                shutil.copy(filename, new_filename)
                
                wf = wave.open(new_filename, 'rb')
                duration = wf.getnframes() / wf.getframerate()
                wf.close()
                
                recording_data = {
                    'name': base_name,
                    'filename': new_filename,
                    'timestamp': datetime.now().isoformat(),
                    'duration': int(duration),
                    'filtered': False,
                    'lowcut': None,
                    'highcut': None,
                    'vosk_transcription': '',
                    'sr_transcription': '',
                    'whisper_transcription': ''
                }
                
                self.recordings.append(recording_data)
                self.save_recordings()
                self.update_recordings_list()
                self.stats_recordings.setText(f"Total Recordings: {len(self.recordings)}")
                
                QMessageBox.information(self, "Success", f"Recording uploaded: {base_name}")
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to upload recording: {str(e)}")
    
    def play_recording(self):
        """Play selected recording"""
        if not hasattr(self, 'selected_recording'):
            QMessageBox.warning(self, "Error", "Please select a recording")
            return
        
        try:
            file_path = self.selected_recording['filename']
            if not os.path.exists(file_path):
                QMessageBox.warning(self, "Error", "Recording file not found")
                return
            
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.media_player.play()
            self.is_playing = True
            self.playback_slider.setEnabled(True)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to play recording: {str(e)}")
    
    def pause_playback(self):
        """Pause audio playback"""
        if self.is_playing:
            self.media_player.pause()
            self.is_playing = False
        else:
            self.media_player.play()
            self.is_playing = True
    
    def stop_playback(self):
        """Stop audio playback"""
        self.media_player.stop()
        self.is_playing = False
        self.playback_slider.setValue(0)
        self.playback_slider.setEnabled(False)
        self.current_time_label.setText("00:00")
        self.total_time_label.setText("00:00")
    
    def on_playback_position_changed(self, position):
        """Update playback position"""
        if not self.playback_slider.isSliderDown():
            duration = self.media_player.duration()
            if duration > 0:
                slider_position = int((position / duration) * 1000)
                self.playback_slider.setValue(slider_position)
        
        self.current_time_label.setText(self.format_time(position))
    
    def on_playback_duration_changed(self, duration):
        """Update total duration"""
        self.total_time_label.setText(self.format_time(duration))
    
    def on_playback_slider_moved(self, position):
        """Handle slider movement"""
        duration = self.media_player.duration()
        if duration > 0:
            new_position = int((position / 1000) * duration)
            self.media_player.setPosition(new_position)
    
    def format_time(self, milliseconds):
        """Format time in milliseconds to MM:SS"""
        seconds = int(milliseconds / 1000)
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:02d}:{secs:02d}"
    
    def transcribe_selected(self):
        """Switch to transcribe tab"""
        if not hasattr(self, 'selected_recording'):
            QMessageBox.warning(self, "Error", "Please select a recording")
            return
        
        self.tabs.setCurrentIndex(2)
    
    def export_recording(self):
        """Export selected recording"""
        if not hasattr(self, 'selected_recording'):
            QMessageBox.warning(self, "Error", "Please select a recording")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Recording", 
            self.selected_recording['name'] + ".wav",
            "WAV Files (*.wav)"
        )
        
        if filename:
            shutil.copy(self.selected_recording['filename'], filename)
            QMessageBox.information(self, "Success", f"Exported to: {filename}")
    
    def delete_recording(self):
        """Delete selected recording"""
        if not hasattr(self, 'selected_recording'):
            QMessageBox.warning(self, "Error", "Please select a recording")
            return
        
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete recording: {self.selected_recording['name']}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if os.path.exists(self.selected_recording['filename']):
                os.remove(self.selected_recording['filename'])
            
            self.recordings = [r for r in self.recordings 
                             if r['filename'] != self.selected_recording['filename']]
            self.save_recordings()
            self.update_recordings_list()
            self.stats_recordings.setText(f"Total Recordings: {len(self.recordings)}")
            
            QMessageBox.information(self, "Success", "Recording deleted")
    
    def visualize_selected(self):
        """Visualize selected recording"""
        if not hasattr(self, 'selected_recording'):
            QMessageBox.warning(self, "Error", "Please select a recording")
            return
        
        if not MATPLOTLIB_AVAILABLE or not SCIPY_AVAILABLE:
            QMessageBox.warning(self, "Error", "Matplotlib and SciPy are required for visualization")
            return
        
        try:
            sample_rate, data = wavfile.read(self.selected_recording['filename'])
            
            self.waveform_canvas.axes.clear()
            time = np.arange(len(data)) / sample_rate
            self.waveform_canvas.axes.plot(time, data, color='#a855f7', linewidth=0.5)
            self.waveform_canvas.axes.set_xlabel('Time (s)', color='white')
            self.waveform_canvas.axes.set_ylabel('Amplitude', color='white')
            self.waveform_canvas.axes.set_title('Waveform', color='white')
            self.waveform_canvas.axes.tick_params(colors='white')
            self.waveform_canvas.axes.grid(True, alpha=0.3)
            self.waveform_canvas.draw()
            
            self.spectrogram_canvas.axes.clear()
            f, t, Sxx = signal.spectrogram(data, sample_rate)
            self.spectrogram_canvas.axes.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                                                    shading='gouraud', cmap='viridis')
            self.spectrogram_canvas.axes.set_ylabel('Frequency (Hz)', color='white')
            self.spectrogram_canvas.axes.set_xlabel('Time (s)', color='white')
            self.spectrogram_canvas.axes.set_title('Spectrogram', color='white')
            self.spectrogram_canvas.axes.tick_params(colors='white')
            self.spectrogram_canvas.draw()
            
            QMessageBox.information(self, "Success", "Visualization complete!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Visualization error: {str(e)}")
    
    def start_transcription(self):
        """Start transcription of selected recording"""
        if not hasattr(self, 'selected_recording'):
            QMessageBox.warning(self, "Error", "Please select a recording first")
            return
        
        methods = []
        if self.vosk_checkbox.isChecked() and VOSK_AVAILABLE:
            methods.append('vosk')
        if self.sr_checkbox.isChecked() and SR_AVAILABLE:
            methods.append('sr')
        if self.whisper_checkbox.isChecked() and WHISPER_AVAILABLE:
            methods.append('whisper')
        
        if not methods:
            QMessageBox.warning(self, "Error", "Please select at least one transcription method")
            return
        
        self.vosk_text.clear()
        self.sr_text.clear()
        self.whisper_text.clear()
        self.transcribe_progress.setValue(0)
        
        QMessageBox.information(self, "Transcription Started", 
                               "Audio will be automatically converted to 16kHz mono for optimal transcription.\n\n" +
                               "This may take a moment depending on the file size and selected methods.")
        
        self.transcribe_thread = TranscriptionThread(
            self.selected_recording['filename'], 
            methods,
            self.selected_vosk_model,
            whisper_model=(self.whisper_model_combo.currentData() if WHISPER_AVAILABLE else 'small'),
            ffmpeg_path=(self.ffmpeg_input.text().strip() if hasattr(self, 'ffmpeg_input') else None)
        )
        self.transcribe_thread.transcription_ready.connect(self.on_transcription_ready)
        self.transcribe_thread.progress_update.connect(self.transcribe_progress.setValue)
        self.transcribe_thread.error_occurred.connect(self.on_error)
        self.transcribe_thread.start()
    
    def on_transcription_ready(self, method, text):
        """Handle completed transcription"""
        if method == 'vosk':
            self.vosk_text.setText(text)
            for rec in self.recordings:
                if rec['filename'] == self.selected_recording['filename']:
                    rec['vosk_transcription'] = text
                    break
        elif method == 'sr':
            self.sr_text.setText(text)
            for rec in self.recordings:
                if rec['filename'] == self.selected_recording['filename']:
                    rec['sr_transcription'] = text
                    break
        elif method == 'whisper':
            self.whisper_text.setText(text)
            for rec in self.recordings:
                if rec['filename'] == self.selected_recording['filename']:
                    rec['whisper_transcription'] = text
                    break
        
        self.save_recordings()
    
    def export_transcription(self, export_type='all'):
        """Export transcription to text file"""
        vosk_text = self.vosk_text.toPlainText()
        sr_text = self.sr_text.toPlainText()
        whisper_text = self.whisper_text.toPlainText()
        
        if not vosk_text and not sr_text and not whisper_text:
            QMessageBox.warning(self, "Error", "No transcription to export")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Transcription",
            "transcription.txt",
            "Text Files (*.txt)"
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                if export_type == 'vosk' or export_type == 'all':
                    if vosk_text:
                        f.write("=" * 60 + "\n")
                        f.write("VOSK TRANSCRIPTION\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(vosk_text + "\n\n")
                
                if export_type == 'sr' or export_type == 'all':
                    if sr_text:
                        f.write("=" * 60 + "\n")
                        f.write("GOOGLE SPEECH RECOGNITION\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(sr_text + "\n\n")
                
                if export_type == 'whisper' or export_type == 'all':
                    if whisper_text:
                        f.write("=" * 60 + "\n")
                        f.write("OPENAI WHISPER\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(whisper_text + "\n")
            
            QMessageBox.information(self, "Success", f"Exported to: {filename}")
    
    def get_stylesheet(self):
        """Get application stylesheet"""
        return """
        QMainWindow {
            background-color: #0f0f1e;
        }
        QWidget {
            background-color: #1a1a2e;
            color: #ffffff;
        }
        QGroupBox {
            border: 2px solid #a855f7;
            border-radius: 8px;
            margin-top: 10px;
            padding: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #a855f7;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #a855f7;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 6px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #9333ea;
        }
        QPushButton:pressed {
            background-color: #7e22ce;
        }
        QPushButton:disabled {
            background-color: #4a4a6a;
            color: #888888;
        }
        QLineEdit, QComboBox, QSpinBox {
            background-color: #0f0f1e;
            border: 2px solid #4a4a6a;
            border-radius: 4px;
            padding: 5px;
            color: white;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
            border-color: #a855f7;
        }
        QProgressBar {
            border: 2px solid #4a4a6a;
            border-radius: 4px;
            text-align: center;
            background-color: #0f0f1e;
        }
        QProgressBar::chunk {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #16a34a, stop:0.5 #eab308, stop:1 #dc2626);
            border-radius: 2px;
        }
        QListWidget {
            background-color: #0f0f1e;
            border: 2px solid #4a4a6a;
            border-radius: 4px;
        }
        QListWidget::item {
            padding: 8px;
            border-bottom: 1px solid #2a2a3e;
        }
        QListWidget::item:selected {
            background-color: #a855f7;
        }
        QTabWidget::pane {
            border: 2px solid #4a4a6a;
            border-radius: 4px;
        }
        QTabBar::tab {
            background-color: #0f0f1e;
            color: #888;
            padding: 10px 20px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #a855f7;
            color: white;
        }
        QTextEdit {
            background-color: #0f0f1e;
            border: 2px solid #4a4a6a;
            border-radius: 4px;
            padding: 10px;
        }
        QCheckBox {
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            border: 2px solid #4a4a6a;
            border-radius: 4px;
            background-color: #0f0f1e;
        }
        QCheckBox::indicator:checked {
            background-color: #a855f7;
            border-color: #a855f7;
        }
        QSlider::groove:horizontal {
            border: 1px solid #4a4a6a;
            height: 8px;
            background: #0f0f1e;
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #a855f7;
            border: 1px solid #a855f7;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        QSlider::handle:horizontal:hover {
            background: #9333ea;
        }
        """
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.is_recording:
            self.stop_recording()
        if self.is_live_listening:
            self.stop_live_listening()
        if self.p:
            self.p.terminate()
        self.media_player.stop()
        event.accept()


def main():
    print("=" * 60)
    print("Starting Laser Microphone Suite")
    print("=" * 60)
    
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        print("Creating main window...")
        window = LaserMicrophoneApp()
        
        print("Showing window...")
        window.show()
        
        print("✓ Application started successfully!")
        print("=" * 60)
        
        sys.exit(app.exec())
    
    except Exception as e:
        print(f"\n{'!' * 60}")
        print(f"FATAL ERROR: {e}")
        print(f"{'!' * 60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()