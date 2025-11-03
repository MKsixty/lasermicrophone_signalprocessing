%% Advanced Speech Enhancement with Adaptive Amplitude Restoration
% Tests multiple algorithms with dynamic amplitude restoration
clear; clc; close all;

%% Load and prepare audio
[x, fs] = audioread('app_audio.wav');
fprintf('File fs = %d Hz\n', fs); 

% Convert to mono
if size(x, 2) > 1
    x = mean(x, 2);
end

% Normalize
x = x / max(abs(x));
audiowrite('stage_0_original.wav', x, fs);

% Calculate original signal statistics
original_rms = rms(x);
original_peak = max(abs(x));
fprintf('Original Signal - RMS: %.4f, Peak: %.4f\n', original_rms, original_peak);

%% STAGE 1: Ultra-Clean Bandpass Filter
fprintf('\n=== Stage 1: Bandpass Filtering ===\n');

f_low = 100;   
f_high = 3400; 

% Very high order for extremely smooth transitions
filter_order = 300;
f_nyq = fs / 2;
f_norm = [f_low f_high] / f_nyq;

% Blackman window (better stopband than Kaiser)
b_bp = fir1(filter_order, f_norm, 'bandpass', blackman(filter_order+1));

% Zero-phase filtering
x_bp = filtfilt(b_bp, 1, x);
audiowrite('stage_1_bandpass.wav', x_bp, fs);
fprintf('✓ Bandpass filter applied: %d-%d Hz\n', f_low, f_high);

%% STAGE 2A: Wiener Filter Denoising
fprintf('\n=== Stage 2A: Wiener Filter Algorithm ===\n');

frame_size = round(0.020 * fs); % 20ms
hop_size = round(0.010 * fs);   % 10ms (50% overlap)
win = hann(frame_size, 'periodic');
nfft = 2^nextpow2(frame_size * 2);

% Estimate noise spectrum from first 0.5 seconds
noise_frames = 25;
noise_power = zeros(nfft, 1);

for i = 1:noise_frames
    idx = (i-1) * hop_size + 1;
    if idx + frame_size - 1 > length(x_bp)
        break;
    end
    
    frame = x_bp(idx:idx+frame_size-1) .* win;
    noise_spec = abs(fft(frame, nfft)).^2;
    noise_power = noise_power + noise_spec;
end

noise_power = noise_power / noise_frames;

% Process all frames with Wiener filter
num_frames = floor((length(x_bp) - frame_size) / hop_size) + 1;
x_wiener = zeros(length(x_bp), 1);
window_sum = zeros(length(x_bp), 1);

for i = 1:num_frames
    idx = (i-1) * hop_size + 1;
    
    if idx + frame_size - 1 > length(x_bp)
        break;
    end
    
    frame = x_bp(idx:idx+frame_size-1) .* win;
    
    % FFT
    frame_fft = fft(frame, nfft);
    frame_power = abs(frame_fft).^2;
    frame_phase = angle(frame_fft);
    
    % Wiener filter gain
    wiener_gain = max(0, 1 - noise_power ./ (frame_power + eps));
    
    % Apply gain smoothing to reduce musical noise
    wiener_gain = smoothdata(wiener_gain, 'gaussian', 5);
    
    % Apply Wiener filter
    filtered_mag = sqrt(frame_power) .* wiener_gain;
    
    % Reconstruct signal
    filtered_fft = filtered_mag .* exp(1j * frame_phase);
    filtered_frame = real(ifft(filtered_fft, nfft));
    filtered_frame = filtered_frame(1:frame_size) .* win;
    
    % Overlap-add
    x_wiener(idx:idx+frame_size-1) = x_wiener(idx:idx+frame_size-1) + filtered_frame;
    window_sum(idx:idx+frame_size-1) = window_sum(idx:idx+frame_size-1) + win.^2;
end

x_wiener = x_wiener ./ max(window_sum, eps);
x_wiener = x_wiener / max(abs(x_wiener)) * 0.95;
audiowrite('stage_2a_wiener_filtered.wav', x_wiener, fs);
fprintf('✓ Wiener filtering complete\n');

%% STAGE 4: Combined Best Approach with Adaptive Restoration
fprintf('\n=== Stage 4: Combined Multi-Stage Enhancement ===\n');

x_combined = x_wiener;

% Apply additional spectral subtraction
frame_size = round(0.020 * fs);
hop_size = round(0.010 * fs);
win = hamming(frame_size, 'periodic');
nfft = 2^nextpow2(frame_size * 2);

% Re-estimate noise from cleaned signal
noise_frames = 25;
noise_spectrum = zeros(nfft, 1);

for i = 1:noise_frames
    idx = (i-1) * hop_size + 1;
    if idx + frame_size - 1 > length(x_combined)
        break;
    end
    
    frame = x_combined(idx:idx+frame_size-1) .* win;
    noise_spec = abs(fft(frame, nfft));
    noise_spectrum = noise_spectrum + noise_spec;
end

noise_spectrum = noise_spectrum / noise_frames;

num_frames = floor((length(x_combined) - frame_size) / hop_size) + 1;
x_final = zeros(length(x_combined), 1);
window_sum = zeros(length(x_combined), 1);

alpha = 1.5;
beta = 0.001;

for i = 1:num_frames
    idx = (i-1) * hop_size + 1;
    
    if idx + frame_size - 1 > length(x_combined)
        break;
    end
    
    frame = x_combined(idx:idx+frame_size-1) .* win;
    frame_fft = fft(frame, nfft);
    frame_mag = abs(frame_fft);
    frame_phase = angle(frame_fft);
    
    cleaned_mag = frame_mag - alpha * noise_spectrum;
    cleaned_mag = max(cleaned_mag, beta * noise_spectrum);
    cleaned_mag = max(cleaned_mag, 0);
    
    cleaned_fft = cleaned_mag .* exp(1j * frame_phase);
    cleaned_frame = real(ifft(cleaned_fft, nfft));
    cleaned_frame = cleaned_frame(1:frame_size) .* win;
    
    x_final(idx:idx+frame_size-1) = x_final(idx:idx+frame_size-1) + cleaned_frame;
    window_sum(idx:idx+frame_size-1) = window_sum(idx:idx+frame_size-1) + win.^2;
end

x_final = x_final ./ max(window_sum, eps);

% Adaptive voice amplification
frame_size = round(0.025 * fs);
hop_size = round(0.010 * fs);
num_frames = floor((length(x_final) - frame_size) / hop_size) + 1;
voice_mask = zeros(length(x_final), 1);

for i = 1:num_frames
    start_idx = (i-1) * hop_size + 1;
    end_idx = start_idx + frame_size - 1;
    
    if end_idx > length(x_final), break; end
    
    frame = x_final(start_idx:end_idx);
    energy = sum(frame.^2);
    zcr = sum(abs(diff(sign(frame)))) / (2 * length(frame));
    
    if zcr > 0.03 && zcr < 0.25 && energy > 0.001
        voice_mask(start_idx:end_idx) = 1;
    end
end

smooth_window = round(0.05 * fs);
voice_mask = conv(voice_mask, ones(smooth_window,1)/smooth_window, 'same');

% Calculate adaptive restoration gain
voice_indices = voice_mask > 0.5;
if sum(voice_indices) > 0
    x_aligned = x;
    if length(x_aligned) > length(x_final)
        x_aligned = x_aligned(1:length(x_final));
    elseif length(x_aligned) < length(x_final)
        x_final_temp = x_final(1:length(x_aligned));
        voice_indices = voice_indices(1:length(x_aligned));
    else
        x_final_temp = x_final;
    end
    
    original_voice_rms = rms(x_aligned(voice_indices));
    denoised_voice_rms = rms(x_final(voice_indices));
    
    if denoised_voice_rms > 0
        combined_gain = original_voice_rms / denoised_voice_rms;
    else
        combined_gain = 1.0;
    end
    
    combined_gain = min(combined_gain, 8.0);
    combined_gain = max(combined_gain, 1.0);
    
    fprintf('Combined approach restoration gain: %.2fx\n', combined_gain);
else
    combined_gain = 2.5;
end

x_final = x_final .* (1 + voice_mask * (combined_gain - 1));

% Hybrid approach: Only consider voice regions for peak calculation
voice_indices_final = voice_mask > 0.5;

if sum(voice_indices_final) > 0
    voice_signal_final = x_final(voice_indices_final);
    
    % Use 99th percentile of voice regions
    sorted_voice_abs = sort(abs(voice_signal_final));
    percentile_idx = round(0.99 * length(sorted_voice_abs));
    peak_voice_99 = sorted_voice_abs(percentile_idx);
    
    % Overall stats for comparison
    sorted_all_abs = sort(abs(x_final));
    percentile_idx_all = round(0.995 * length(sorted_all_abs));
    peak_all_99p5 = sorted_all_abs(percentile_idx_all);
    absolute_peak = max(abs(x_final));
    
    fprintf('Combined - Voice 99th: %.4f, Overall 99.5th: %.4f, Absolute: %.4f\n', ...
            peak_voice_99, peak_all_99p5, absolute_peak);
    
    % Normalize to voice peak
    x_final = x_final / peak_voice_99 * 0.98;
    
    % Soft clip using tanh
    x_final = tanh(x_final * 1.05) * 0.99;
else
    % Fallback to standard percentile approach
    fprintf('Combined - No voice detected, using percentile approach\n');
    sorted_abs = sort(abs(x_final));
    percentile_idx = round(0.995 * length(sorted_abs));
    peak_99p5 = sorted_abs(percentile_idx);
    
    x_final = x_final / peak_99p5 * 0.99;
    x_final(x_final > 0.99) = 0.99;
    x_final(x_final < -0.99) = -0.99;
end

fprintf('Final combined RMS: %.4f (%.1f%% of original)\n', rms(x_final), (rms(x_final)/original_rms)*100);
fprintf('Final combined peak: %.4f\n', max(abs(x_final)));

audiowrite('stage_4_combined.wav', x_final, fs);
fprintf('✓ Combined enhancement complete (Whisper-optimized)\n');

%% Quality Metrics
fprintf('\n=== Quality Metrics ===\n');

noise_orig = std(x(1:min(round(0.5*fs), length(x))));
noise_bp = std(x_bp(1:min(round(0.5*fs), length(x_bp))));
noise_wiener = std(x_wiener(1:min(round(0.5*fs), length(x_wiener))));
noise_final = std(x_final(1:min(round(0.5*fs), length(x_final))));

fprintf('Noise Floor (std dev):\n');
fprintf('  Original:              %.6f\n', noise_orig);
fprintf('  Bandpass:              %.6f (%.1f%% reduction)\n', noise_bp, (1-noise_bp/noise_orig)*100);

fprintf('\n=== Files Generated ===\n');
fprintf('Stage 0: stage_0_original.wav\n');
fprintf('Stage 1: stage_1_bandpass.wav\n');
fprintf('Stage 2: Multiple denoising algorithms\n');
fprintf('  - stage_2a_wiener_filtered.wav\n');
fprintf('Stage 4: stage_4_combined.wav\n');

fprintf('\n✓ Processing complete! Adaptive restoration maintains original amplitude.\n');