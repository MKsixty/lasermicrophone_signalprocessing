# MATLAB Integration Guide for Laser Microphone App

## Overview
The Laser Microphone App now supports automatic MATLAB post-processing of all recorded audio files using your `final_filter.m` script.

## Features
- ✅ Automatic MATLAB processing after each recording
- ✅ Stores both original and processed audio files
- ✅ Easy toggle between original and processed versions
- ✅ All playback, export, and visualization functions work with both versions
- ✅ Clear visual indicators showing which recordings have been processed

## Setup

### 1. Ensure MATLAB is Installed

The app uses **subprocess** to call MATLAB directly - **no Python packages needed!**

**Requirements:**
- MATLAB R2024b or later installed on your system
- MATLAB executable accessible (automatically detected in common locations)

**Common MATLAB Installation Paths:**
- `C:\Program Files\MATLAB\R2025b\bin\matlab.exe`
- `C:\Program Files\MATLAB\R2025a\bin\matlab.exe`
- `C:\Program Files\MATLAB\R2024b\bin\matlab.exe`

The app will automatically detect MATLAB in these locations.

### 2. Place Your MATLAB Script

Ensure that `final_filter.m` is in the same directory as your Python app (`laser_microphoneV1.py`).

The script expects:
- **Input:** `app_audio.wav` (automatically created by the app)
- **Output:** `stage_4_combined.wav` (the final processed audio)

## Usage

### Recording with MATLAB Processing

1. **Enable MATLAB Processing:**
   - In the left panel under "Recording", check the "MATLAB Post-Processing" checkbox
   - This checkbox is automatically enabled if MATLAB Engine is available

2. **Record Audio:**
   - Enter a recording name
   - Configure your filters (optional)
   - Click "Start Recording"
   - Click "Stop Recording" when done

3. **Automatic Processing:**
   - The app will automatically process your recording through MATLAB
   - You'll see a progress dialog showing the processing status
   - Both original and processed files are saved

### Viewing/Playing Recordings

**Toggle Between Versions:**
- In the "Recordings" tab, use the dropdown at the top:
  - 🎵 **Original** - Shows/plays the raw recorded audio
  - ✨ **MATLAB Processed** - Shows/plays the MATLAB-enhanced audio

**Visual Indicators:**
- Recordings that have been processed show "✨ MATLAB" in the list

### Functions That Support Both Versions

All these functions automatically use the selected version (original/processed):

- ▶ **Play** - Plays the selected version
- 💾 **Export** - Exports the selected version
- 📊 **Visualize** - Visualizes the selected version

**Note:** Transcription always uses the original recording file, regardless of the version toggle setting.

### File Naming Convention

- **Original:** `RecordingName_timestamp.wav`
- **Processed:** `RecordingName_timestamp_processed.wav`

## MATLAB Script Requirements

Your `final_filter.m` script should:

1. Read audio from `app_audio.wav` in the same directory
2. Process the audio (bandpass filtering, denoising, etc.)
3. Save the final output as `stage_4_combined.wav`

The current script already does this correctly!

## Troubleshooting

### "MATLAB not found" error
- Ensure MATLAB is properly installed on your system
- Verify MATLAB.exe exists in one of the common installation paths
- Add MATLAB bin directory to your system PATH (optional)
- Restart the app after installation

### MATLAB processing fails
- Check that `final_filter.m` is in the correct directory
- Verify the script runs successfully in MATLAB directly
- Check the error message in the dialog for specific issues

### Processed file not found
- Ensure your MATLAB script saves to `stage_4_combined.wav`
- Check MATLAB console for any script errors
- Verify file permissions in the recordings directory

## Benefits of MATLAB Processing

Using the `final_filter.m` script provides:
- **Advanced Wiener filtering** for noise reduction
- **Spectral subtraction** for cleaner audio
- **Adaptive amplitude restoration** maintaining original signal strength
- **Voice-optimized processing** for better speech recognition
- **Professional-grade audio enhancement** beyond basic filtering

## Disabling MATLAB Processing

If you want to skip MATLAB processing for a recording:
- Simply uncheck the "MATLAB Post-Processing" checkbox before recording
- The app will only save the original recording
- You can still apply filters during recording if needed

## Performance Notes

- MATLAB processing takes 10-30 seconds depending on recording length
- Processing happens in the background - you can continue using the app
- The progress dialog shows current processing status
- Longer recordings will take proportionally longer to process

