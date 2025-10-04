# app.py
import os
import sys
import threading
import queue
import time
import glob
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import sounddevice as sd
import soundfile as sf
import numpy as np

# ---- CONFIG ----
CAPTURE_DIR = Path('captures')
PROCESSED_DIR = Path('processed')
CAPTURE_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
DEFAULT_FS = 44100
CHANNELS = 1
DTYPE = 'float32'

# MATLAB engine: try import
USE_MATLAB_ENGINE = False
try:
    import matlab.engine
    USE_MATLAB_ENGINE = True
except Exception:
    USE_MATLAB_ENGINE = False

# ---- Recording helper ----
rec_q = queue.Queue()
rec_event = threading.Event()
recording_thread = None

def record_to_file(filename, duration=None, fs=DEFAULT_FS):
    """Records until stopped (if duration is None) or for duration seconds."""
    frames = []

    def callback(indata, frames_count, time_info, status):
        if status:
            print("Recording status:", status, file=sys.stderr)
        rec_q.put(indata.copy())

    with sd.InputStream(samplerate=fs, channels=CHANNELS, dtype=DTYPE, callback=callback):
        print("Recording started. Press Stop to finish.")
        start = time.time()
        while not rec_event.is_set():
            try:
                data = rec_q.get(timeout=0.1)
                frames.append(data)
            except queue.Empty:
                pass
            if duration and (time.time() - start) >= duration:
                break

    if len(frames) == 0:
        print("No audio captured.")
        return False

    all_data = np.concatenate(frames, axis=0)
    if all_data.ndim > 1:
        all_data = np.mean(all_data, axis=1)  # to mono
    sf.write(str(filename), all_data, fs)
    print(f"Saved recording to {filename}")
    return True

# ---- Playback helper ----
def play_file(filename):
    data, fs = sf.read(str(filename), always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    sd.play(data, fs)
    sd.wait()

# ---- MATLAB call helpers ----
def call_matlab_process(input_wav, output_wav, params=None, eng=None):
    """
    If eng (MATLAB engine) provided, call engine.process_file(...)
    Otherwise call MATLAB via subprocess using -batch (requires matlab on PATH).
    """
    if params is None:
        params = {}  # keep default behavior in MATLAB

    if USE_MATLAB_ENGINE and eng is not None:
        # Convert params dict to MATLAB struct via engine
        try:
            # Build MATLAB struct in Python by creating a dict of fieldnames
            # (the engine will convert python dict -> matlab struct in recent engines)
            eng.process_file(str(input_wav), str(output_wav), params, nargout=0)
            return True, "OK (engine)"
        except Exception as e:
            return False, str(e)
    else:
        # Use subprocess to run matlab in batch mode. Create a small one-liner that calls the function
        # Escape paths properly for MATLAB
        input_wav_m = str(input_wav).replace('\\','\\')
        output_wav_m = str(output_wav).replace('\\','\\')
        # Build matlab command (batch) - ensure function is on MATLAB path
        matlab_cmd = f"try, process_file('{input_wav_m}','{output_wav_m}'), catch ME, disp(getReport(ME)), exit(1), end, exit(0);"
        # Note: On Windows, use "matlab.exe -batch" or "-nodisplay -nosplash -r"
        # Simpler approach: write a small .m script and call "matlab -batch myscript"
        # We'll write a temporary script
        tmp_script = Path('tmp_process_script.m')
        tmp_script.write_text(f"process_file('{input_wav_m}','{output_wav_m}');\n")
        try:
            # Use -batch if available (R2019a+). Otherwise fallback to -r
            proc = subprocess.run(['matlab', '-batch', f"run('{tmp_script.resolve()}')"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
            os.remove(tmp_script)
            if proc.returncode != 0:
                return False, proc.stdout + proc.stderr
            return True, "OK (subprocess)"
        except FileNotFoundError:
            return False, "MATLAB executable not found on PATH. Install MATLAB or use MATLAB Engine."
        except subprocess.TimeoutExpired:
            return False, "MATLAB call timed out."

# ---- GUI ----
class App:
    def __init__(self, root):
        self.root = root
        root.title("Python + MATLAB Audio Capture & DSP")
        self.listbox = tk.Listbox(root, width=60)
        self.listbox.grid(row=0, column=0, columnspan=4, padx=8, pady=8)

        self.refresh_btn = tk.Button(root, text="Refresh list", command=self.refresh_list)
        self.refresh_btn.grid(row=1, column=0, padx=4, pady=4)

        self.record_btn = tk.Button(root, text="Record", command=self.start_record)
        self.record_btn.grid(row=1, column=1, padx=4, pady=4)

        self.stop_btn = tk.Button(root, text="Stop", command=self.stop_record, state='disabled')
        self.stop_btn.grid(row=1, column=2, padx=4, pady=4)

        self.play_btn = tk.Button(root, text="Play selected", command=self.play_selected)
        self.play_btn.grid(row=1, column=3, padx=4, pady=4)

        self.process_btn = tk.Button(root, text="Process selected (MATLAB)", command=self.process_selected)
        self.process_btn.grid(row=2, column=0, padx=4, pady=4)

        self.play_processed_btn = tk.Button(root, text="Play processed", command=self.play_processed)
        self.play_processed_btn.grid(row=2, column=1, padx=4, pady=4)

        self.batch_btn = tk.Button(root, text="Process all (MATLAB)", command=self.process_all)
        self.batch_btn.grid(row=2, column=2, padx=4, pady=4)

        self.quit_btn = tk.Button(root, text="Quit", command=root.quit)
        self.quit_btn.grid(row=2, column=3, padx=4, pady=4)

        self.status = tk.Label(root, text="Ready", anchor='w')
        self.status.grid(row=3, column=0, columnspan=4, sticky='we', padx=8, pady=4)

        self.refresh_list()

        self.matlab_eng = None
        if USE_MATLAB_ENGINE:
            try:
                self.status['text'] = 'Starting MATLAB engine...'
                root.update()
                self.matlab_eng = matlab.engine.start_matlab()
                self.status['text'] = 'MATLAB engine connected.'
            except Exception as e:
                self.status['text'] = 'MATLAB engine failed: ' + str(e)

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        files = sorted(CAPTURE_DIR.glob('*.wav'))
        for f in files:
            self.listbox.insert(tk.END, str(f))
        self.status['text'] = f'{len(files)} capture(s) found.'

    def start_record(self):
        fname = simpledialog.askstring("Filename", "Enter filename (without extension):", parent=self.root)
        if not fname:
            return
        self.current_fname = CAPTURE_DIR / (fname + '.wav')
        rec_event.clear()
        self.record_btn['state'] = 'disabled'
        self.stop_btn['state'] = 'normal'
        self.status['text'] = 'Recording...'
        # start thread
        t = threading.Thread(target=self._record_thread, args=(self.current_fname,))
        t.daemon = True
        t.start()

    def _record_thread(self, filename):
        try:
            record_to_file(filename)
            self.status['text'] = f'Recorded {filename}'
        except Exception as e:
            self.status['text'] = f'Record failed: {e}'
        finally:
            self.record_btn['state'] = 'normal'
            self.stop_btn['state'] = 'disabled'
            self.refresh_list()

    def stop_record(self):
        rec_event.set()
        self.status['text'] = 'Stopping...'

    def play_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("Select", "Please select a capture file to play.")
            return
        fname = Path(self.listbox.get(sel[0]))
        threading.Thread(target=self._play_thread, args=(fname,)).start()

    def _play_thread(self, fname):
        self.status['text'] = f'Playing {fname}'
        try:
            play_file(fname)
        except Exception as e:
            self.status['text'] = f'Play error: {e}'
        finally:
            self.status['text'] = 'Ready'

    def process_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("Select", "Please select a capture file to process.")
            return
        infile = Path(self.listbox.get(sel[0]))
        outfile = PROCESSED_DIR / ('processed_' + infile.name)
        threading.Thread(target=self._process_thread, args=(infile, outfile)).start()

    def _process_thread(self, infile, outfile):
        self.status['text'] = f'Processing {infile} ...'
        self._disable_buttons()
        try:
            # params can be passed as a dict
            params = {
                'fs': DEFAULT_FS,
                'bp': [100, 4500],
                'filt_order': 512,
                'window_len': 1024,
                'hop': 256,
                'noise_frames': 6,
                'beta': 0.02,
                'alpha': 4,
                'gate_thresh_db': -20,
                'smooth_win': 3
            }
            if USE_MATLAB_ENGINE and self.matlab_eng is not None:
                ok, msg = True, 'engine'
                try:
                    # MATLAB engine will convert python dict -> matlab struct in many setups
                    self.matlab_eng.process_file(str(infile), str(outfile), params, nargout=0)
                except Exception as e:
                    ok, msg = False, str(e)
            else:
                ok, msg = call_matlab_process(infile, outfile)
            if ok:
                self.status['text'] = f'Processed -> {outfile}'
            else:
                self.status['text'] = f'Processing failed: {msg}'
        except Exception as e:
            self.status['text'] = 'Processing error: ' + str(e)
        finally:
            self._enable_buttons()
            self.refresh_list()

    def play_processed(self):
        proc_files = sorted(PROCESSED_DIR.glob('processed_*.wav'))
        if not proc_files:
            messagebox.showinfo("No files", "No processed files found.")
            return
        # play first processed file
        threading.Thread(target=self._play_thread, args=(proc_files[0],)).start()

    def process_all(self):
        files = sorted(CAPTURE_DIR.glob('*.wav'))
        if not files:
            messagebox.showinfo("No files", "No capture WAVs found.")
            return
        threading.Thread(target=self._process_all_thread, args=(files,)).start()

    def _process_all_thread(self, files):
        self._disable_buttons()
        for infile in files:
            outfile = PROCESSED_DIR / ('processed_' + infile.name)
            self.status['text'] = f'Processing {infile.name} ...'
            try:
                if USE_MATLAB_ENGINE and self.matlab_eng is not None:
                    self.matlab_eng.process_file(str(infile), str(outfile), nargout=0)
                else:
                    ok, msg = call_matlab_process(infile, outfile)
                    if not ok:
                        self.status['text'] = f'Failed: {msg}'
                        break
            except Exception as e:
                self.status['text'] = f'Error: {e}'
                break
        self.status['text'] = 'Batch processing done.'
        self._enable_buttons()
        self.refresh_list()

    def _disable_buttons(self):
        self.record_btn['state'] = 'disabled'
        self.stop_btn['state'] = 'disabled'
        self.play_btn['state'] = 'disabled'
        self.process_btn['state'] = 'disabled'
        self.batch_btn['state'] = 'disabled'
        self.play_processed_btn['state'] = 'disabled'

    def _enable_buttons(self):
        self.record_btn['state'] = 'normal'
        self.play_btn['state'] = 'normal'
        self.process_btn['state'] = 'normal'
        self.batch_btn['state'] = 'normal'
        self.play_processed_btn['state'] = 'normal'


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()

