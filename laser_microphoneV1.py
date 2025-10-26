"""
🎤 Laser Microphone Suite - Advanced Audio Processing Application
Enhanced Features:
- 48kHz, 16-bit, Mono WAV recording
- Real-time bandpass filtering (80-3400 Hz)
- Live listening with filtering and volume control (1-10x amplification)
- Audio visualization (waveform, spectrogram)
- Multiple Vosk model support with selection
- Side-by-side transcription (Vosk + Speech Recognition + Whisper)
- Automatic 16kHz conversion for transcription
- Recording upload functionality
- Working playback feature with timeline
- Recording history management
"""

import sys
import os
import json
import traceback
import subprocess
import tempfile
import time


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
                                  QProgressBar, QSpinBox, QGroupBox, QListWidgetItem,
                                  QScrollArea, QSizePolicy, QFrame, QGridLayout)
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QUrl
    from PyQt6.QtGui import QFont, QColor
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    print(" PyQt6 loaded successfully")
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
    print(" PyAudio loaded successfully")
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
    print(" SciPy loaded successfully")
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: SciPy not available. Install with: pip install scipy")

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
    print(" Matplotlib loaded successfully")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: Matplotlib not available. Install with: pip install matplotlib")

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
    print(" Vosk loaded successfully")
except ImportError:
    VOSK_AVAILABLE = False
    print("WARNING: Vosk not available. Install with: pip install vosk")

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
    print(" SpeechRecognition loaded successfully")
except ImportError:
    SR_AVAILABLE = False
    print("WARNING: speech_recognition not available")

try:
    import whisper
    WHISPER_AVAILABLE = True
    print(" OpenAI Whisper loaded successfully")
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
        'accuracy': ''
    },
    'vosk-model-en-us-0.22-lgraph': {
        'name': 'vosk-model-en-us-0.22-lgraph',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip',
        'size': '128 MB',
        'description': 'Medium English model - Balanced speed and accuracy',
        'accuracy': ''
    },
    'vosk-model-en-us-0.22': {
        'name': 'vosk-model-en-us-0.22',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip',
        'size': '1.8 GB',
        'description': 'Large English model - Best accuracy',
        'accuracy': ''
    }
}


class BandpassFilter:
    """Real-time bandpass filter implementation"""
    def __init__(self, lowcut=80, highcut=3400, fs=48000, order=5):
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
    
    def set_filter(self, enabled, lowcut=80, highcut=3400):
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
        self.output_device = None  # Will be set later
        self.sample_rate = sample_rate
        self.chunk = 1024
        self.is_running = False
        self.filter_enabled = False
        self.bandpass_filter = None
        self.volume_multiplier = 1.0
    
    def set_filter(self, enabled, lowcut=80, highcut=3400):
        self.filter_enabled = enabled
        if enabled and SCIPY_AVAILABLE:
            self.bandpass_filter = BandpassFilter(lowcut, highcut, self.sample_rate)
    
    def set_volume(self, multiplier):
        """Set volume amplification (1-30x)"""
        self.volume_multiplier = max(1.0, min(30.0, multiplier))
    
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
                output_device_index=self.output_device,
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
                print(f" Converted audio to 16kHz using ffmpeg: {temp_file}")
                return temp_file
            else:
                print(" ffmpeg not found, using scipy for conversion...")
                
                if not SCIPY_AVAILABLE:
                    print(" scipy not available, using original file")
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
                print(f" Converted audio to 16kHz using scipy: {temp_file}")
                return temp_file
                
        except Exception as e:
            print(f"Audio conversion error: {e}, using original file")
            return input_file
    
    def transcribe_vosk(self):
        """📝 Transcribe using Vosk offline"""
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
                print(f" Warning: Vosk expects 16kHz audio, got {wf.getframerate()}Hz")
            
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
        """📝 Transcribe using Python Speech Recognition - Simple and reliable"""
        try:
            recognizer = sr.Recognizer()
            # Adjust energy threshold for better speech detection (lower = more sensitive)
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = False
            
            self.progress_update.emit(33)
            
            # Priority: Try original file first, then converted file
            files_to_try = [self.audio_file]
            if self.converted_file and self.converted_file != self.audio_file and os.path.exists(self.converted_file):
                files_to_try.append(self.converted_file)
            
            audio_data = None
            used_file = None
            
            # Try each file in order
            for audio_file in files_to_try:
                try:
                    print(f"[SR] Trying file: {audio_file}")
                    with sr.AudioFile(audio_file) as source:
                        # Record the ENTIRE audio file (duration=-1 or not specified means all)
                        audio_data = recognizer.record(source, duration=None)
                        used_file = audio_file
                        print(f"[SR] Successfully loaded entire audio from: {audio_file}")
                        break
                except Exception as e:
                    print(f"[SR] Failed to read {audio_file}: {e}")
                    continue
            
            if audio_data is None:
                self.error_occurred.emit("SpeechRecognition: Could not read audio file. Ensure it's a valid WAV file.")
                return
            
            print(f"[SR] Starting Google Speech Recognition...")
            self.progress_update.emit(66)
            
            # Simple transcription with retry logic
            def split_audio_data(audio_data, chunk_duration=30):
                """Split audio data into chunks"""
                try:
                    # Get audio data properties
                    frame_rate = audio_data.sample_rate
                    frame_count = len(audio_data.frame_data) // 2  # 16-bit audio = 2 bytes per sample
                    samples_per_chunk = int(frame_rate * chunk_duration)
                    
                    chunks = []
                    for i in range(0, frame_count, samples_per_chunk):
                        chunk = sr.AudioData(
                            audio_data.frame_data[i*2:(i+samples_per_chunk)*2],
                            frame_rate,
                            audio_data.sample_width
                        )
                        chunks.append(chunk)
                    return chunks
                except Exception as e:
                    print(f"[SR] Error splitting audio: {e}")
                    return [audio_data]  # Return original as single chunk if splitting fails
            
            max_attempts = 3
            for attempt in range(max_attempts):
                # Update progress based on attempt (33% per attempt)
                self.progress_update.emit(66 + (attempt * 11))  # Start at 66%, increment by 11% per attempt
                
                try:
                    print(f"[SR] Recognition attempt {attempt + 1}/{max_attempts}")
                    # Try to get all possible transcriptions
                    result = recognizer.recognize_google(audio_data, show_all=True)
                    
                    if result and isinstance(result, dict) and 'alternative' in result:
                        # Get the best transcription (highest confidence)
                        text = result['alternative'][0]['transcript']
                        print(f"[SR] Success! Transcription: {text[:100]}...")
                        self.progress_update.emit(100)
                        self.transcription_ready.emit('sr', text)
                        return
                    elif result and isinstance(result, list) and result:
                        # Handle case where we get a list of alternatives
                        text = result[0]  # Take best match
                        print(f"[SR] Partial success! Best match: {text[:100]}...")
                        self.progress_update.emit(100)
                        self.transcription_ready.emit('sr', text)
                        return
                    else:
                        raise sr.UnknownValueError()
                    
                except sr.UnknownValueError:
                    print(f"[SR] Could not understand audio clearly (attempt {attempt + 1}/{max_attempts})")
                    if attempt == max_attempts - 1:
                        # On final attempt, try chunking the audio
                        try:
                            print("[SR] Attempting chunked transcription...")
                            chunks = split_audio_data(audio_data)
                            transcriptions = []
                            
                            for i, chunk in enumerate(chunks):
                                try:
                                    chunk_text = recognizer.recognize_google(chunk, show_all=False)
                                    if chunk_text:
                                        transcriptions.append(chunk_text)
                                        print(f"[SR] Chunk {i+1}/{len(chunks)} transcribed successfully")
                                    else:
                                        print(f"[SR] Chunk {i+1}/{len(chunks)} was unclear")
                                except sr.UnknownValueError:
                                    print(f"[SR] Chunk {i+1}/{len(chunks)} was unclear")
                                except sr.RequestError as e:
                                    print(f"[SR] Error transcribing chunk {i+1}: {e}")
                                    time.sleep(2)  # Brief pause between chunks on error
                            
                            if transcriptions:
                                full_text = " ".join(transcriptions)
                                print(f"[SR] Assembled chunked transcription: {full_text[:100]}...")
                                self.progress_update.emit(100)
                                self.transcription_ready.emit('sr', f"[Chunked transcription]: {full_text}")
                                return
                            else:
                                raise sr.UnknownValueError()
                                
                        except Exception as e:
                            print(f"[SR] Chunked transcription failed: {e}")
                            self.progress_update.emit(100)
                            self.transcription_ready.emit('sr', "[Could not understand audio clearly after multiple attempts]")
                            return
                        except:
                            self.progress_update.emit(100)
                            self.transcription_ready.emit('sr', "[Could not understand audio clearly after multiple attempts]")
                            return
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
                except sr.RequestError as e:
                    print(f"[SR] Request error (attempt {attempt + 1}/{max_attempts}): {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        error_msg = (
                            f"Speech Recognition network error: {e}\n\n"
                            "Possible causes:\n"
                            " No internet connection\n"
                            " Google API rate limit\n"
                            " Firewall blocking request"
                        )
                        self.error_occurred.emit(error_msg)
                        return
                        
                except Exception as ex:
                    print(f"[SR] Unexpected error: {ex}")
                    self.error_occurred.emit(f"Speech Recognition error: {ex}")
                    return
            
        except Exception as e:
            error_msg = f"Speech Recognition error: {str(e)}"
            print(f"[SR] {error_msg}")
            self.error_occurred.emit(error_msg)

    def transcribe_whisper(self):
        """📝 Transcribe using OpenAI Whisper"""
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
            
            # Enable responsive resizing
            fig.set_tight_layout(True)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding
            )
            self.updateGeometry()
        
        def resizeEvent(self, event):
            super().resizeEvent(event)
            self.figure.tight_layout()  # Adjust layout on resize


class LaserMicrophoneApp(QMainWindow):
    def __init__(self):
        super().__init__()
        print("Initializing 🎤 Laser Microphone App...")
        
        self.setWindowTitle("🎤 Laser Microphone Suite - Advanced Audio Processing")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(self.get_stylesheet())
        
        # Initialize recordings first
        self.recordings_dir = "recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
        self.recordings = []  # Initialize empty list
        print("Loading recordings...")
        self.recordings = self.load_recordings()  # Load recordings from disk
        print(f"Loaded {len(self.recordings)} recordings")
        
        self.input_devices = []
        self.output_devices = []
        self.selected_input_device = 0
        self.selected_output_device = 0
        
        if PYAUDIO_AVAILABLE:
            try:
                self.p = pyaudio.PyAudio()
                self.input_devices, self.output_devices = self.get_audio_devices()
                print(f" Found {len(self.input_devices)} input devices and {len(self.output_devices)} output devices")
                
                # Auto-select first USB device for input if available
                usb_input = next((dev for dev in self.input_devices if dev['is_usb']), None)
                if usb_input:
                    self.selected_input_device = usb_input['index']
                    print(f" Auto-selected USB input device: {usb_input['name']}")
                
                # Auto-select first non-USB device for output if available
                default_output = next((dev for dev in self.output_devices if not dev['is_usb']), None)
                if default_output:
                    self.selected_output_device = default_output['index']
                    print(f" Auto-selected default output device: {default_output['name']}")
                
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
        
        self.filter_enabled = True
        self.lowcut = 80
        self.highcut = 3400
        self.volume_multiplier = 1.0
        
        self.selected_vosk_model = 'vosk-model-small-en-us-0.15'
        
        try:
            self.init_ui()
            print(" UI initialized successfully")
        except Exception as e:
            print(f"ERROR initializing UI: {e}")
            traceback.print_exc()
        
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_time)
        self.recording_duration = 0
        
        print(" Application initialized successfully")
    
    def get_audio_devices(self):
        """Get list of available audio input and output devices"""
        input_devices = []
        output_devices = []
        if not self.p:
            return input_devices, output_devices
        
        try:
            for i in range(self.p.get_device_count()):
                info = self.p.get_device_info_by_index(i)
                device_info = {
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate']),
                    'is_usb': 'usb' in info['name'].lower()
                }
                
                if info['maxInputChannels'] > 0:
                    input_devices.append(device_info)
                if info['maxOutputChannels'] > 0:
                    output_devices.append(device_info)
                    
                # Print device info for debugging
                print(f"Found device {i}: {info['name']}")
                print(f"  Input channels: {info['maxInputChannels']}")
                print(f"  Output channels: {info['maxOutputChannels']}")
                print(f"  Sample rate: {info['defaultSampleRate']}")
                
        except Exception as e:
            print(f"Error getting audio devices: {e}")
        
        # Sort devices to prefer USB devices for input and default devices for output
        input_devices.sort(key=lambda x: (not x['is_usb'], x['name']))
        output_devices.sort(key=lambda x: (x['is_usb'], x['name']))  # Non-USB first for output
        
        return input_devices, output_devices
    
    def load_recordings(self):
        """Load recordings from database file"""
        db_file = os.path.join(self.recordings_dir, "recordings.json")
        if os.path.exists(db_file):
            try:
                with open(db_file, 'r') as f:
                    recordings = json.load(f)
                    # Verify each recording file exists
                    valid_recordings = []
                    for rec in recordings:
                        if os.path.exists(rec['filename']):
                            valid_recordings.append(rec)
                        else:
                            print(f"Warning: Recording file not found: {rec['filename']}")
                    return valid_recordings
            except Exception as e:
                print(f"Error loading recordings: {e}")
                return []
        return []
    
    def save_recordings(self):
        """Save recordings to database file"""
        db_file = os.path.join(self.recordings_dir, "recordings.json")
        try:
            # Create recordings directory if it doesn't exist
            os.makedirs(self.recordings_dir, exist_ok=True)
            with open(db_file, 'w') as f:
                json.dump(self.recordings, f, indent=2)
            print(f"Saved {len(self.recordings)} recordings to {db_file}")
        except Exception as e:
            print(f"Error saving recordings: {e}")
            QMessageBox.warning(self, "Error", f"Failed to save recordings: {str(e)}")
        self.update_recordings_list()  # Refresh the UI after saving
    
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Create and configure left panel
        left_panel = self.create_left_panel()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(400)
        main_layout.addWidget(left_panel)
        
        # Create and configure right panel with scroll area
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        right_panel = self.create_right_panel()
        right_scroll.setWidget(right_panel)
        
        main_layout.addWidget(right_scroll, 1)  # Give right panel more stretch
        
        # Set minimum window size
        self.setMinimumSize(1000, 700)
    
    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        header = QLabel(" 🎤 Laser Microphone")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header.setStyleSheet("color: #a855f7; padding: 10px;")
        layout.addWidget(header)
        
        device_group = QGroupBox("Audio Devices")
        device_layout = QVBoxLayout()
        
        # Input device selection
        input_label = QLabel("Recording Input Device:")
        device_layout.addWidget(input_label)
        self.input_device_combo = QComboBox()
        
        # Output device selection
        output_label = QLabel("Playback Output Device:")
        device_layout.addWidget(output_label)
        self.output_device_combo = QComboBox()
        
        self.input_devices, self.output_devices = self.get_audio_devices()
        
        if self.input_devices:
            for device in self.input_devices:
                # Clean up device name to prevent display issues
                device_name = device['name']
                # Remove or replace problematic characters
                device_name = device_name.replace('\x00', '').replace('\r', '').replace('\n', ' ')
                # Truncate very long names
                if len(device_name) > 50:
                    device_name = device_name[:47] + "..."
                
                self.input_device_combo.addItem(
                    f"{'[USB] ' if device['is_usb'] else ''}{device_name} ({device['sample_rate']} Hz)", 
                    device['index']
                )
            # Auto-select first USB device if available
            usb_index = next((i for i, d in enumerate(self.input_devices) if d['is_usb']), 0)
            self.input_device_combo.setCurrentIndex(usb_index)
            self.input_device_combo.currentIndexChanged.connect(self.on_input_device_changed)
        else:
            self.input_device_combo.addItem("No input devices found", -1)
            self.input_device_combo.setEnabled(False)
        
        if self.output_devices:
            for device in self.output_devices:
                # Clean up device name to prevent display issues
                device_name = device['name']
                # Remove or replace problematic characters
                device_name = device_name.replace('\x00', '').replace('\r', '').replace('\n', ' ')
                # Truncate very long names
                if len(device_name) > 50:
                    device_name = device_name[:47] + "..."
                
                self.output_device_combo.addItem(
                    f"{'[USB] ' if device['is_usb'] else ''}{device_name} ({device['sample_rate']} Hz)", 
                    device['index']
                )
            # Auto-select first non-USB device if available
            default_index = next((i for i, d in enumerate(self.output_devices) if not d['is_usb']), 0)
            self.output_device_combo.setCurrentIndex(default_index)
            self.output_device_combo.currentIndexChanged.connect(self.on_output_device_changed)
        else:
            self.output_device_combo.addItem("No output devices found", -1)
            self.output_device_combo.setEnabled(False)
        
        device_layout.addWidget(self.input_device_combo)
        device_layout.addWidget(output_label)
        device_layout.addWidget(self.output_device_combo)
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
        self.volume_slider.setMinimum(10)  # 1.0x minimum
        self.volume_slider.setMaximum(300)  # 30.0x maximum
        self.volume_slider.setValue(10)  # Default 1.0x
        self.volume_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.volume_slider.setTickInterval(50)  # Ticks every 5.0x
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        
        volume_warning = QLabel("High amplification may cause distortion")
        volume_warning.setStyleSheet("color: #ffa500; font-size: 10px;")
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(volume_warning)
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
        
        self.record_btn = QPushButton(" 🔴 Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setMinimumHeight(50)
        if not PYAUDIO_AVAILABLE:
            self.record_btn.setEnabled(False)
            self.record_btn.setText("PyAudio Required")
        recording_layout.addWidget(self.record_btn)
        
        self.live_btn = QPushButton(" 🔊 Start Live Listening")
        self.live_btn.clicked.connect(self.toggle_live_listening)
        self.live_btn.setMinimumHeight(40)
        if not PYAUDIO_AVAILABLE:
            self.live_btn.setEnabled(False)
            self.live_btn.setText("PyAudio Required")
        recording_layout.addWidget(self.live_btn)
        
        recording_group.setLayout(recording_layout)
        layout.addWidget(recording_group)
        
        stats_group = QGroupBox("📊 Session Stats")
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(2)  # Reduce spacing between items
        stats_layout.setContentsMargins(8, 8, 8, 8)  # Reduce margins
        
        # Recording stats in a more compact format
        self.stats_recordings = QLabel(f" Total 📁 Recordings: {len(self.recordings)}")
        self.stats_recordings.setStyleSheet("font-size: 11px; font-weight: bold; color: #a855f7;")
        stats_layout.addWidget(self.stats_recordings)
        
        # Audio format info in one line
        format_info = QLabel(" 🎵 48 kHz • 16-bit • WAV Mono")
        format_info.setStyleSheet("font-size: 10px; color: #888;")
        stats_layout.addWidget(format_info)
        
        # Library status in a more compact format
        status_text = " 📚 Libraries: "
        status_text += f"PyAudio:{'' if PYAUDIO_AVAILABLE else ''} "
        status_text += f"SciPy:{'' if SCIPY_AVAILABLE else ''} "
        status_text += f"Matplotlib:{'' if MATPLOTLIB_AVAILABLE else ''} "
        status_text += f"Vosk:{'' if VOSK_AVAILABLE else ''} "
        status_text += f"SR:{'' if SR_AVAILABLE else ''} "
        status_text += f"Whisper:{'' if WHISPER_AVAILABLE else ''}"
        
        # FFmpeg availability on PATH (initial check)
        ff_on_path = is_ffmpeg_available(None)
        status_text += f" FFmpeg:{'' if ff_on_path else ''}"
        
        self.ffmpeg_status_label = QLabel(status_text)
        self.ffmpeg_status_label.setStyleSheet("font-size: 9px; color: #666; line-height: 1.2;")
        self.ffmpeg_status_label.setWordWrap(True)
        stats_layout.addWidget(self.ffmpeg_status_label)
        
        stats_group.setLayout(stats_layout)
        stats_group.setMaximumHeight(120)  # Limit the height of the stats group
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """Create right content panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)
        
        # Create and add tabs
        recordings_tab = self.create_recordings_tab()
        self.tabs.addTab(recordings_tab, " 📁 Recordings")
        
        visualize_tab = self.create_visualize_tab()
        self.tabs.addTab(visualize_tab, " 📊 Visualize")
        
        transcribe_tab = self.create_transcribe_tab()
        self.tabs.addTab(transcribe_tab, " 📝 Transcribe")
        
        layout.addWidget(self.tabs)
        
        return panel
    
    def create_recordings_tab(self):
        """Create recordings list tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Recordings list with scroll area - takes up most of the space
        list_container = QWidget()
        list_layout = QVBoxLayout(list_container)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(5)
        
        # Add a header for the recordings list
        recordings_header = QLabel("📁 Recorded Audio Files")
        recordings_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #a855f7; padding: 5px;")
        list_layout.addWidget(recordings_header)
        
        self.recordings_list = QListWidget()
        self.recordings_list.itemClicked.connect(self.on_recording_selected)
        self.recordings_list.setMinimumHeight(400)  # Increased height for better visibility
        self.update_recordings_list()
        list_layout.addWidget(self.recordings_list)
        
        layout.addWidget(list_container)
        
        # Playback controls - positioned above buttons
        playback_container = QWidget()
        playback_layout = QVBoxLayout(playback_container)
        playback_layout.setContentsMargins(5, 5, 5, 5)
        playback_layout.setSpacing(5)
        
        # Add playback header
        playback_header = QLabel("🎵 Playback Controls")
        playback_header.setStyleSheet("font-size: 12px; font-weight: bold; color: #a855f7; padding: 2px;")
        playback_layout.addWidget(playback_header)
        
        time_layout = QHBoxLayout()
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setSpacing(8)
        
        self.current_time_label = QLabel("00:00")
        self.current_time_label.setStyleSheet("color: #a855f7; font-weight: bold; font-size: 11px;")
        self.current_time_label.setMinimumWidth(45)
        time_layout.addWidget(self.current_time_label)
        
        self.playback_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_slider.setMinimum(0)
        self.playback_slider.setMaximum(1000)
        self.playback_slider.setValue(0)
        self.playback_slider.sliderMoved.connect(self.on_playback_slider_moved)
        self.playback_slider.setEnabled(False)
        self.playback_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #4a4a6a;
                height: 6px;
                background: #0f0f1e;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #a855f7;
                border: 1px solid #a855f7;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #9333ea;
            }
        """)
        time_layout.addWidget(self.playback_slider)
        
        self.total_time_label = QLabel("00:00")
        self.total_time_label.setStyleSheet("color: #888; font-size: 11px;")
        self.total_time_label.setMinimumWidth(45)
        time_layout.addWidget(self.total_time_label)
        
        playback_layout.addLayout(time_layout)
        layout.addWidget(playback_container)
        
        # Control buttons in one horizontal line
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        btn_layout.setContentsMargins(5, 5, 5, 5)
        
        upload_btn = QPushButton("📤 Upload")
        upload_btn.clicked.connect(self.upload_recording)
        upload_btn.setMinimumHeight(35)
        btn_layout.addWidget(upload_btn)
        
        play_btn = QPushButton("▶ Play")
        play_btn.clicked.connect(self.play_recording)
        play_btn.setMinimumHeight(35)
        btn_layout.addWidget(play_btn)
        
        pause_btn = QPushButton("⏸ Pause")
        pause_btn.clicked.connect(self.pause_playback)
        pause_btn.setMinimumHeight(35)
        btn_layout.addWidget(pause_btn)
        
        stop_btn = QPushButton("⏹ Stop")
        stop_btn.clicked.connect(self.stop_playback)
        stop_btn.setMinimumHeight(35)
        btn_layout.addWidget(stop_btn)
        
        export_btn = QPushButton("💾 Export")
        export_btn.clicked.connect(self.export_recording)
        export_btn.setMinimumHeight(35)
        btn_layout.addWidget(export_btn)
        
        delete_btn = QPushButton("🗑 Delete")
        delete_btn.clicked.connect(self.delete_recording)
        delete_btn.setMinimumHeight(35)
        btn_layout.addWidget(delete_btn)
        
        transcribe_btn = QPushButton("📝 Transcribe Selected Recording")
        transcribe_btn.clicked.connect(self.transcribe_selected)
        transcribe_btn.setMinimumHeight(35)
        btn_layout.addWidget(transcribe_btn)
        
        layout.addLayout(btn_layout)
        
        return tab
    
    def create_visualize_tab(self):
        """Create visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        if MATPLOTLIB_AVAILABLE and SCIPY_AVAILABLE:
            waveform_group = QGroupBox("Waveform Visualization")
            waveform_layout = QVBoxLayout()
            
            self.waveform_canvas = MplCanvas(self, width=8, height=3, dpi=100)
            self.waveform_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            waveform_layout.addWidget(self.waveform_canvas)
            waveform_group.setLayout(waveform_layout)
            layout.addWidget(waveform_group, 2)  # Higher stretch factor
            
            spectrogram_group = QGroupBox("Spectrogram Analysis")
            spectrogram_layout = QVBoxLayout()
            
            self.spectrogram_canvas = MplCanvas(self, width=8, height=3, dpi=100)
            self.spectrogram_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            spectrogram_layout.addWidget(self.spectrogram_canvas)
            spectrogram_group.setLayout(spectrogram_layout)
            layout.addWidget(spectrogram_group, 2)  # Higher stretch factor
            
            control_group = QGroupBox("Visualization Controls")
            control_layout = QVBoxLayout()
            
            load_btn = QPushButton("Load Selected Recording for Visualization")
            load_btn.setMinimumHeight(40)
            load_btn.clicked.connect(self.visualize_selected)
            control_layout.addWidget(load_btn)
            
            control_group.setLayout(control_layout)
            layout.addWidget(control_group, 1)  # Lower stretch factor
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
            # Update status label in 📊 Session Stats if present
            try:
                ff_on_path = is_ffmpeg_available(None)
                if hasattr(self, 'ffmpeg_status_label'):
                    prev = self.ffmpeg_status_label.text().split('\n')
                    # replace last line with FFmpeg status
                    if prev:
                        prev[-1] = f" FFmpeg on PATH: {'' if ff_on_path else ''}"
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
        
        start_transcribe_btn = QPushButton(" 🎙 Start Transcription")
        start_transcribe_btn.clicked.connect(self.start_transcription)
        btn_layout.addWidget(start_transcribe_btn)
        
        export_vosk_btn = QPushButton(" 💾 💾 Export Vosk")
        export_vosk_btn.clicked.connect(lambda: self.export_transcription('vosk'))
        btn_layout.addWidget(export_vosk_btn)
        
        export_sr_btn = QPushButton(" 💾 💾 Export Google SR")
        export_sr_btn.clicked.connect(lambda: self.export_transcription('sr'))
        btn_layout.addWidget(export_sr_btn)
        
        export_whisper_btn = QPushButton(" 💾 💾 Export Whisper")
        export_whisper_btn.clicked.connect(lambda: self.export_transcription('whisper'))
        btn_layout.addWidget(export_whisper_btn)
        
        export_all_btn = QPushButton(" 💾 💾 Export All")
        export_all_btn.clicked.connect(lambda: self.export_transcription('all'))
        btn_layout.addWidget(export_all_btn)
        
        layout.addLayout(btn_layout)
        
        return tab
    
    def on_input_device_changed(self, index):
        """Handle input device selection change"""
        if index >= 0 and self.input_devices:
            device = self.input_devices[index]
            self.selected_input_device = device['index']
            print(f"Selected input device: {device['name']}")
            
            # Update recorders if running
            if self.is_recording:
                self.stop_recording()
            if self.is_live_listening:
                self.stop_live_listening()
    
    def on_output_device_changed(self, index):
        """Handle output device selection change"""
        if index >= 0 and self.output_devices:
            device = self.output_devices[index]
            self.selected_output_device = device['index']
            print(f"Selected output device: {device['name']}")
            
            # Update live processor if running
            if self.is_live_listening:
                self.stop_live_listening()
                time.sleep(0.1)  # Brief pause to ensure clean restart
                self.start_live_listening()
    
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
            
            self.recorder = AudioRecorder(self.selected_input_device)
            self.recorder.set_filter(self.filter_enabled, self.lowcut, self.highcut)
            self.recorder.filename = filename
            self.recorder.audio_data_ready.connect(self.on_audio_data)
            self.recorder.recording_finished.connect(self.on_recording_finished)
            self.recorder.error_occurred.connect(self.on_error)
            self.recorder.start()
            
            self.is_recording = True
            self.record_btn.setText(" ⏹ Stop Recording")
            self.record_btn.setStyleSheet("background-color: #dc2626;")
            self.name_input.setEnabled(False)
            
            self.recording_duration = 0
            self.recording_timer.start(1000)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start recording: {str(e)}")
    
    def stop_recording(self):
        """⏹ Stop audio recording"""
        if self.recorder:
            self.recorder.stop()
        
        self.is_recording = False
        self.record_btn.setText(" 🔴 Start Recording")
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
            self.live_processor = LiveAudioProcessor(self.selected_input_device)
            self.live_processor.set_filter(self.filter_enabled, self.lowcut, self.highcut)
            self.live_processor.output_device = self.selected_output_device  # Set output device
            self.live_processor.set_volume(self.volume_multiplier)
            self.live_processor.audio_level.connect(self.update_volume_level)
            self.live_processor.start()
            
            self.is_live_listening = True
            self.live_btn.setText(" ⏹ Stop Live Listening")
            self.live_btn.setStyleSheet("background-color: #16a34a;")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start live listening: {str(e)}")
    
    def stop_live_listening(self):
        """⏹ Stop live audio monitoring"""
        if self.live_processor:
            self.live_processor.stop()
            self.live_processor.wait()
        
        self.is_live_listening = False
        self.live_btn.setText(" 🔊 Start Live Listening")
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
        if not os.path.exists(filename):
            QMessageBox.warning(self, "Error", f"Recording file not found: {filename}")
            return
            
        recording_data = {
            'name': self.name_input.text().strip() or 'Untitled Recording',
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
        print(f"Added new recording: {recording_data['name']}")
        self.save_recordings()  # This will also update the list
        self.stats_recordings.setText(f"Total 📁 Recordings: {len(self.recordings)}")
        
        QMessageBox.information(self, "Success", f"Recording saved: {filename}")
        self.name_input.clear()
    
    def on_error(self, error_msg):
        """Handle errors"""
        QMessageBox.critical(self, "Error", error_msg)
    
    def update_recordings_list(self):
        """Update recordings list widget"""
        if not hasattr(self, 'recordings_list'):
            print("Warning: recordings_list not initialized")
            return
            
        print(f"Updating recordings list with {len(self.recordings)} recordings")
        self.recordings_list.clear()
        
        # Debug: Print current recordings
        print("Current recordings:")
        for rec in self.recordings:
            print(f"- {rec.get('name', 'Unnamed')} ({rec.get('filename', 'No file')})")
        
        self.recordings_list.setStyleSheet("""
            QListWidget {
                background-color: #0f0f1e;
                border: 2px solid #4a4a6a;
                border-radius: 8px;
                padding: 8px;
                font-size: 12px;
            }
            QListWidget::item {
                background-color: #1a1a2e;
                border: 1px solid #2a2a3e;
                border-radius: 6px;
                padding: 12px;
                margin: 3px;
                min-height: 45px;
                font-weight: 500;
            }
            QListWidget::item:selected {
                background-color: #a855f7;
                border: 2px solid #9333ea;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #2a2a3e;
                border: 1px solid #4a4a6a;
            }
        """)
        # Add recordings to list widget
        try:
            for rec in reversed(self.recordings):
                if not isinstance(rec, dict) or 'filename' not in rec:
                    print(f"Warning: Invalid recording data: {rec}")
                    continue
                
                # Fix file path - normalize backslashes to forward slashes
                filename = rec['filename'].replace('\\', '/')
                
                # Check if file exists with normalized path
                if not os.path.exists(filename):
                    print(f"Warning: Recording file not found: {filename}")
                    continue
                
                # Format recording name with better styling
                name = rec.get('name', 'Unnamed')
                if len(name) > 25:
                    name = name[:22] + "..."
                
                # Format date nicely
                timestamp = rec.get('timestamp', '')
                if timestamp:
                    try:
                        date_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        date_part = date_obj.strftime("%Y-%m-%d")
                    except:
                        date_part = timestamp[:10] if len(timestamp) >= 10 else "Unknown"
                else:
                    date_part = "Unknown"
                
                # Format duration nicely
                duration = rec.get('duration', 0)
                if duration >= 60:
                    mins = duration // 60
                    secs = duration % 60
                    duration_part = f"{mins}m {secs}s"
                else:
                    duration_part = f"{duration}s"
                
                # Create formatted item text - all on one line
                item_text = f"🎵 {name} • 📅 {date_part} • ⏱ {duration_part}"
                
                if rec.get('filtered', False):
                    lowcut = rec.get('lowcut', 0)
                    highcut = rec.get('highcut', 0)
                    item_text += f" • 🔊 {lowcut}-{highcut}Hz"
                
                item = QListWidgetItem(item_text)
                # Store the original recording data with normalized filename
                rec_copy = rec.copy()
                rec_copy['filename'] = filename
                item.setData(Qt.ItemDataRole.UserRole, rec_copy)
                self.recordings_list.addItem(item)
                print(f"Added item to list: {item_text}")
            
            print(f"Updated list widget with {self.recordings_list.count()} items")
        except Exception as e:
            print(f"Error updating recordings list: {e}")
            import traceback
            traceback.print_exc()
    
    def on_recording_selected(self, item):
        """Handle recording selection"""
        self.selected_recording = item.data(Qt.ItemDataRole.UserRole)
    
    def upload_recording(self):
        """📤 Upload an external recording"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "📤 Upload Recording", 
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
                self.stats_recordings.setText(f"Total 📁 Recordings: {len(self.recordings)}")
                
                QMessageBox.information(self, "Success", f"Recording uploaded: {base_name}")
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to upload recording: {str(e)}")
    
    def play_recording(self):
        """▶ Play selected recording"""
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
        """⏸ Pause audio playback"""
        if self.is_playing:
            self.media_player.pause()
            self.is_playing = False
        else:
            self.media_player.play()
            self.is_playing = True
    
    def stop_playback(self):
        """⏹ Stop audio playback"""
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
        """💾 Export selected recording"""
        if not hasattr(self, 'selected_recording'):
            QMessageBox.warning(self, "Error", "Please select a recording")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "💾 Export Recording", 
            self.selected_recording['name'] + ".wav",
            "WAV Files (*.wav)"
        )
        
        if filename:
            shutil.copy(self.selected_recording['filename'], filename)
            QMessageBox.information(self, "Success", f"💾 Exported to: {filename}")
    
    def delete_recording(self):
        """🗑 Delete selected recording"""
        if not hasattr(self, 'selected_recording'):
            QMessageBox.warning(self, "Error", "Please select a recording")
            return
        
        reply = QMessageBox.question(
            self, "Confirm 🗑 Delete",
            f"🗑 Delete recording: {self.selected_recording['name']}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if os.path.exists(self.selected_recording['filename']):
                os.remove(self.selected_recording['filename'])
            
            self.recordings = [r for r in self.recordings 
                             if r['filename'] != self.selected_recording['filename']]
            self.save_recordings()
            self.update_recordings_list()
            self.stats_recordings.setText(f"Total 📁 Recordings: {len(self.recordings)}")
            
            QMessageBox.information(self, "Success", "Recording deleted")
    
    def visualize_selected(self):
        """📊 Visualize selected recording"""
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
        """💾 Export transcription to text file"""
        vosk_text = self.vosk_text.toPlainText()
        sr_text = self.sr_text.toPlainText()
        whisper_text = self.whisper_text.toPlainText()
        
        if not vosk_text and not sr_text and not whisper_text:
            QMessageBox.warning(self, "Error", "No transcription to export")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "💾 Export Transcription",
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
            
            QMessageBox.information(self, "Success", f"💾 Exported to: {filename}")
    
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
    print("Starting 🎤 Laser Microphone Suite")
    print("=" * 60)
    
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        print("Creating main window...")
        window = LaserMicrophoneApp()
        
        print("Showing window...")
        window.show()
        
        print(" Application started successfully!")
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