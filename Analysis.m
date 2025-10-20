%% Advanced Speech Enhancement with Wiener Filter
% Tests multiple algorithms BEFORE amplification to find best buzz removal
clear; clc; close all;

%% Load and prepare audio
[x, fs] = audioread('Final.wav');
fprintf('File fs = %d Hz\n', fs); 

% Convert to mono
if size(x, 2) > 1
    x = mean(x, 2);
end

% Normalize
x = x / max(abs(x));
audiowrite('stage_0_original.wav', x, fs);

%% STAGE 1: Ultra-Clean Bandpass Filter
fprintf('\n=== Stage 1: Bandpass Filtering ===\n');

f_low = 300;   
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
    % H(f) = max(0, 1 - noise_power / signal_power)
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

%% STAGE 2B: Spectral Subtraction with Musical Noise Suppression
fprintf('\n=== Stage 2B: Enhanced Spectral Subtraction ===\n');

frame_size = round(0.020 * fs);
hop_size = round(0.010 * fs);
win = hamming(frame_size, 'periodic');
nfft = 2^nextpow2(frame_size * 2);

% Estimate noise from first 0.5 seconds
noise_frames = 25;
noise_spectrum = zeros(nfft, 1);

for i = 1:noise_frames
    idx = (i-1) * hop_size + 1;
    if idx + frame_size - 1 > length(x_bp)
        break;
    end
    
    frame = x_bp(idx:idx+frame_size-1) .* win;
    noise_spec = abs(fft(frame, nfft));
    noise_spectrum = noise_spectrum + noise_spec;
end

noise_spectrum = noise_spectrum / noise_frames;

% Process all frames
num_frames = floor((length(x_bp) - frame_size) / hop_size) + 1;
x_ss = zeros(length(x_bp), 1);
window_sum = zeros(length(x_bp), 1);

% Adaptive over-subtraction factor
alpha = 2.0;  % Over-subtraction factor
beta = 0.001; % Spectral floor (prevents musical noise)

for i = 1:num_frames
    idx = (i-1) * hop_size + 1;
    
    if idx + frame_size - 1 > length(x_bp)
        break;
    end
    
    frame = x_bp(idx:idx+frame_size-1) .* win;
    
    % FFT
    frame_fft = fft(frame, nfft);
    frame_mag = abs(frame_fft);
    frame_phase = angle(frame_fft);
    
    % Spectral subtraction with over-subtraction
    cleaned_mag = frame_mag - alpha * noise_spectrum;
    
    % Apply spectral floor (reduces musical noise)
    cleaned_mag = max(cleaned_mag, beta * noise_spectrum);
    
    % Half-wave rectification (alternative to hard floor)
    cleaned_mag = max(cleaned_mag, 0);
    
    % Reconstruct
    cleaned_fft = cleaned_mag .* exp(1j * frame_phase);
    cleaned_frame = real(ifft(cleaned_fft, nfft));
    cleaned_frame = cleaned_frame(1:frame_size) .* win;
    
    % Overlap-add
    x_ss(idx:idx+frame_size-1) = x_ss(idx:idx+frame_size-1) + cleaned_frame;
    window_sum(idx:idx+frame_size-1) = window_sum(idx:idx+frame_size-1) + win.^2;
end

x_ss = x_ss ./ max(window_sum, eps);
x_ss = x_ss / max(abs(x_ss)) * 0.95;
audiowrite('stage_2b_spectral_subtraction.wav', x_ss, fs);
fprintf('✓ Spectral subtraction complete\n');

%% STAGE 2C: Adaptive Spectral Gating (AI-Inspired Approach)
fprintf('\n=== Stage 2C: Adaptive Spectral Gating ===\n');

% This mimics AI behavior by learning noise characteristics adaptively
frame_size = round(0.020 * fs);
hop_size = round(0.010 * fs);
win = hann(frame_size, 'periodic');
nfft = 2^nextpow2(frame_size * 2);

% Learn noise profile from multiple segments (like AI training)
noise_learning_duration = min(1.0, length(x_bp)/fs * 0.2); % 1 sec or 20% of audio
noise_frames = round(noise_learning_duration * fs / hop_size);

% Estimate noise statistics with variance
noise_mean = zeros(nfft, 1);
noise_var = zeros(nfft, 1);

for i = 1:noise_frames
    idx = (i-1) * hop_size + 1;
    if idx + frame_size - 1 > length(x_bp), break; end
    
    frame = x_bp(idx:idx+frame_size-1) .* win;
    frame_mag = abs(fft(frame, nfft));
    
    noise_mean = noise_mean + frame_mag;
    noise_var = noise_var + frame_mag.^2;
end

noise_mean = noise_mean / noise_frames;
noise_var = (noise_var / noise_frames) - noise_mean.^2;
noise_std = sqrt(max(noise_var, eps));

% Process with adaptive threshold (AI-inspired)
num_frames = floor((length(x_bp) - frame_size) / hop_size) + 1;
x_adaptive = zeros(length(x_bp), 1);
window_sum = zeros(length(x_bp), 1);

% Adaptive parameters (learned from data)
sensitivity = 2.5; % Threshold sensitivity
smoothing_frames = 3;
gate_buffer = zeros(nfft, smoothing_frames);
buffer_idx = 1;

for i = 1:num_frames
    idx = (i-1) * hop_size + 1;
    if idx + frame_size - 1 > length(x_bp), break; end
    
    frame = x_bp(idx:idx+frame_size-1) .* win;
    frame_fft = fft(frame, nfft);
    frame_mag = abs(frame_fft);
    frame_phase = angle(frame_fft);
    
    % Adaptive gate: speech-likely regions get gain > 1, noise gets < 1
    snr_estimate = (frame_mag - noise_mean) ./ (noise_std + eps);
    
    % Soft gating function (sigmoid-like, mimics neural activation)
    gate = 1 ./ (1 + exp(-sensitivity * (snr_estimate - 1)));
    
    % Temporal smoothing (like RNN memory)
    gate_buffer(:, buffer_idx) = gate;
    buffer_idx = mod(buffer_idx, smoothing_frames) + 1;
    
    if i > smoothing_frames
        gate = mean(gate_buffer, 2);
    end
    
    % Apply gate
    enhanced_mag = frame_mag .* gate;
    
    % Reconstruct
    enhanced_fft = enhanced_mag .* exp(1j * frame_phase);
    enhanced_frame = real(ifft(enhanced_fft, nfft));
    enhanced_frame = enhanced_frame(1:frame_size) .* win;
    
    x_adaptive(idx:idx+frame_size-1) = x_adaptive(idx:idx+frame_size-1) + enhanced_frame;
    window_sum(idx:idx+frame_size-1) = window_sum(idx:idx+frame_size-1) + win.^2;
end

x_adaptive = x_adaptive ./ max(window_sum, eps);
x_adaptive = x_adaptive / max(abs(x_adaptive)) * 0.95;
audiowrite('stage_2c_adaptive_gating.wav', x_adaptive, fs);
fprintf('✓ Adaptive spectral gating complete\n');

%% STAGE 2D: Median Filter (Removes Impulse Noise)
fprintf('\n=== Stage 2D: Median Filtering ===\n');

% Apply median filter to remove impulse noise/clicks
med_window = round(0.003 * fs); % 3ms window
if mod(med_window, 2) == 0
    med_window = med_window + 1; % Must be odd
end

x_med = medfilt1(x_bp, med_window);
x_med = x_med / max(abs(x_med)) * 0.95;
audiowrite('stage_2d_median_filtered.wav', x_med, fs);
fprintf('✓ Median filtering complete\n');

%% STAGE 3: Voice Amplification (Applied to BEST denoised version)
fprintf('\n=== Stage 3: Voice Amplification (2.5x) ===\n');

% Apply to all denoised versions
denoised_versions = {x_wiener, x_ss, x_adaptive, x_med};
version_names = {'Wiener', 'SpectralSub', 'Adaptive', 'Median'};

amplified_versions = cell(4, 1);

for v = 1:4
    signal = denoised_versions{v};
    
    frame_size = round(0.025 * fs);
    hop_size = round(0.010 * fs);
    
    num_frames = floor((length(signal) - frame_size) / hop_size) + 1;
    voice_mask = zeros(length(signal), 1);
    
    % Voice Activity Detection
    for i = 1:num_frames
        start_idx = (i-1) * hop_size + 1;
        end_idx = start_idx + frame_size - 1;
        
        if end_idx > length(signal)
            break;
        end
        
        frame = signal(start_idx:end_idx);
        
        % Voice detection
        energy = sum(frame.^2);
        zcr = sum(abs(diff(sign(frame)))) / (2 * length(frame));
        
        is_voice = (zcr > 0.03 && zcr < 0.25 && energy > 0.001);
        
        if is_voice
            voice_mask(start_idx:end_idx) = 1;
        end
    end
    
    % Smooth mask
    smooth_window = round(0.05 * fs);
    voice_mask = conv(voice_mask, ones(smooth_window,1)/smooth_window, 'same');
    
    % Amplify voice regions by 2.5x
    voice_gain = 2.5;
    amplified = signal .* (1 + voice_mask * (voice_gain - 1));
    amplified = amplified / max(abs(amplified)) * 0.95;
    
    amplified_versions{v} = amplified;
    
    filename = sprintf('stage_3_%s_amplified.wav', lower(version_names{v}));
    audiowrite(filename, amplified, fs);
    fprintf('✓ %s amplified (2.5x)\n', version_names{v});
end

%% STAGE 4: Combined Best Approach (Wiener + Spectral Subtraction)
fprintf('\n=== Stage 4: Combined Multi-Stage Enhancement ===\n');

% Apply Wiener filter first, then spectral subtraction for extra cleaning
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

alpha = 1.5;  % Gentler over-subtraction on already cleaned signal
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

% Voice amplification
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

x_final = x_final .* (1 + voice_mask * 1.5);
x_final = x_final / max(abs(x_final)) * 0.95;

audiowrite('stage_4_combined_best.wav', x_final, fs);
fprintf('✓ Combined enhancement complete\n');

%% Generate Clean Reference Signal
fprintf('\n=== Generating Clean Reference Signal ===\n');

% Synthesize a clean speech-like signal for comparison
duration = min(5, length(x)/fs); % 5 seconds or length of input
t_clean = 0:1/fs:duration-1/fs;

% Create multi-formant speech-like signal
% Formants for vowel /a/: F1=700Hz, F2=1200Hz, F3=2500Hz
f0 = 150; % Fundamental frequency (male voice)

% Generate harmonic series
clean_signal = zeros(size(t_clean));
for harmonic = 1:20
    freq = harmonic * f0;
    if freq < fs/2
        % Amplitude rolloff for higher harmonics
        amplitude = 1 / harmonic^1.2;
        clean_signal = clean_signal + amplitude * sin(2*pi*freq*t_clean);
    end
end

% Add formant structure (resonances)
% Formant 1 at 700 Hz
[b1, a1] = butter(2, [650 750]/(fs/2), 'bandpass');
formant1 = filter(b1, a1, clean_signal);

% Formant 2 at 1200 Hz
[b2, a2] = butter(2, [1100 1300]/(fs/2), 'bandpass');
formant2 = filter(b2, a2, clean_signal) * 0.6;

% Formant 3 at 2500 Hz
[b3, a3] = butter(2, [2300 2700]/(fs/2), 'bandpass');
formant3 = filter(b3, a3, clean_signal) * 0.3;

% Combine formants
clean_signal = formant1 + formant2 + formant3;

% Add natural amplitude modulation (like speech prosody)
mod_freq = 4; % 4 Hz modulation (syllable rate)
amplitude_envelope = 0.5 + 0.5 * sin(2*pi*mod_freq*t_clean);
clean_signal = clean_signal .* amplitude_envelope;

% Normalize
clean_signal = clean_signal / max(abs(clean_signal)) * 0.7;
clean_signal = clean_signal(:);

% Save clean reference
audiowrite('reference_clean_speech.wav', clean_signal, fs);
fprintf('✓ Clean reference signal generated\n');

%% Visualization with Clean Reference
fprintf('\n=== Generating Visualizations ===\n');

% Main comparison figure
figure('Position', [50 50 1600 1200]);

signals = {x, x_bp, x_wiener, x_ss, x_adaptive, x_med, amplified_versions{3}, x_final};
titles = {'Original', 'Bandpass', 'Wiener Filtered', 'Spectral Sub', ...
          'Adaptive Gating', 'Median Filtered', 'Adaptive + Amp (2.5x)', 'Combined Best'};

for i = 1:8
    sig = signals{i};
    t = (0:length(sig)-1) / fs;
    
    % Waveform
    subplot(8, 2, 2*i-1);
    plot(t, sig, 'LineWidth', 0.5);
    title(titles{i}, 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on; axis tight; ylim([-1 1]);
    
    % Spectrogram
    subplot(8, 2, 2*i);
    spectrogram(sig, hann(512), 256, 1024, fs, 'yaxis');
    title([titles{i} ' - Spectrogram']);
    colormap jet; caxis([-80 0]);
end

sgtitle('Multi-Algorithm Speech Enhancement with AI-Inspired Adaptive Gating', 'FontSize', 14, 'FontWeight', 'bold');

%% Clean Reference Comparison Figure
figure('Position', [100 100 1600 800]);
sgtitle('Comparison with Clean Reference Signal', 'FontSize', 14, 'FontWeight', 'bold');

% Clean Reference
subplot(3, 3, 1);
plot(t_clean, clean_signal, 'g', 'LineWidth', 0.5);
title('Clean Reference - Waveform', 'FontWeight', 'bold', 'Color', 'g');
xlabel('Time (s)'); ylabel('Amplitude');
grid on; axis tight; ylim([-1 1]);

subplot(3, 3, 2);
spectrogram(clean_signal, hann(512), 256, 1024, fs, 'yaxis');
title('Clean Reference - Spectrogram', 'FontWeight', 'bold');
colormap jet; caxis([-80 0]);
ylim([0 4]);

subplot(3, 3, 3);
[pxx_clean, f_clean] = pwelch(clean_signal, hann(2048), 1024, 4096, fs);
plot(f_clean, 10*log10(pxx_clean), 'g', 'LineWidth', 1.5);
title('Clean Reference - PSD', 'FontWeight', 'bold');
xlabel('Frequency (Hz)'); ylabel('Power (dB/Hz)');
grid on; xlim([0 4000]);

% Original (Noisy)
subplot(3, 3, 4);
t_orig = (0:length(x)-1) / fs;
plot(t_orig, x, 'r', 'LineWidth', 0.5);
title('Original (Noisy) - Waveform', 'FontWeight', 'bold', 'Color', 'r');
xlabel('Time (s)'); ylabel('Amplitude');
grid on; axis tight; ylim([-1 1]);

subplot(3, 3, 5);
spectrogram(x, hann(512), 256, 1024, fs, 'yaxis');
title('Original (Noisy) - Spectrogram', 'FontWeight', 'bold');
colormap jet; caxis([-80 0]);
ylim([0 4]);

subplot(3, 3, 6);
[pxx_orig, f_orig] = pwelch(x, hann(2048), 1024, 4096, fs);
plot(f_orig, 10*log10(pxx_orig), 'r', 'LineWidth', 1.5);
title('Original (Noisy) - PSD', 'FontWeight', 'bold');
xlabel('Frequency (Hz)'); ylabel('Power (dB/Hz)');
grid on; xlim([0 4000]);

% Best Enhanced (Combined)
subplot(3, 3, 7);
t_final = (0:length(x_final)-1) / fs;
plot(t_final, x_final, 'b', 'LineWidth', 0.5);
title('Enhanced (Combined Best) - Waveform', 'FontWeight', 'bold', 'Color', 'b');
xlabel('Time (s)'); ylabel('Amplitude');
grid on; axis tight; ylim([-1 1]);

subplot(3, 3, 8);
spectrogram(x_final, hann(512), 256, 1024, fs, 'yaxis');
title('Enhanced (Combined Best) - Spectrogram', 'FontWeight', 'bold');
colormap jet; caxis([-80 0]);
ylim([0 4]);

subplot(3, 3, 9);
[pxx_final, f_final] = pwelch(x_final, hann(2048), 1024, 4096, fs);
plot(f_final, 10*log10(pxx_final), 'b', 'LineWidth', 1.5);
title('Enhanced (Combined Best) - PSD', 'FontWeight', 'bold');
xlabel('Frequency (Hz)'); ylabel('Power (dB/Hz)');
grid on; xlim([0 4000]);

%% Spectral Comparison Figure
figure('Position', [150 150 1400 600]);
sgtitle('Power Spectral Density Comparison', 'FontSize', 14, 'FontWeight', 'bold');

% Calculate PSDs for all versions
[pxx_bp, f_bp] = pwelch(x_bp, hann(2048), 1024, 4096, fs);
[pxx_wiener, f_wiener] = pwelch(x_wiener, hann(2048), 1024, 4096, fs);
[pxx_ss, f_ss] = pwelch(x_ss, hann(2048), 1024, 4096, fs);
[pxx_adaptive, f_adaptive] = pwelch(x_adaptive, hann(2048), 1024, 4096, fs);

subplot(2, 2, 1);
hold on;
plot(f_clean, 10*log10(pxx_clean), 'g', 'LineWidth', 2, 'DisplayName', 'Clean Reference');
plot(f_orig, 10*log10(pxx_orig), 'r', 'LineWidth', 1.5, 'DisplayName', 'Original (Noisy)');
hold off;
title('Clean vs Noisy', 'FontWeight', 'bold');
xlabel('Frequency (Hz)'); ylabel('Power (dB/Hz)');
legend('Location', 'northeast'); grid on; xlim([0 4000]);

subplot(2, 2, 2);
hold on;
plot(f_clean, 10*log10(pxx_clean), 'g', 'LineWidth', 2, 'DisplayName', 'Clean Reference');
plot(f_wiener, 10*log10(pxx_wiener), 'b', 'LineWidth', 1.5, 'DisplayName', 'Wiener Filtered');
hold off;
title('Wiener Filter vs Clean', 'FontWeight', 'bold');
xlabel('Frequency (Hz)'); ylabel('Power (dB/Hz)');
legend('Location', 'northeast'); grid on; xlim([0 4000]);

subplot(2, 2, 3);
hold on;
plot(f_clean, 10*log10(pxx_clean), 'g', 'LineWidth', 2, 'DisplayName', 'Clean Reference');
plot(f_adaptive, 10*log10(pxx_adaptive), 'm', 'LineWidth', 1.5, 'DisplayName', 'Adaptive Gating');
hold off;
title('Adaptive Gating vs Clean', 'FontWeight', 'bold');
xlabel('Frequency (Hz)'); ylabel('Power (dB/Hz)');
legend('Location', 'northeast'); grid on; xlim([0 4000]);

subplot(2, 2, 4);
hold on;
plot(f_clean, 10*log10(pxx_clean), 'g', 'LineWidth', 2, 'DisplayName', 'Clean Reference');
plot(f_final, 10*log10(pxx_final), 'k', 'LineWidth', 1.5, 'DisplayName', 'Combined Best');
hold off;
title('Combined Best vs Clean', 'FontWeight', 'bold');
xlabel('Frequency (Hz)'); ylabel('Power (dB/Hz)');
legend('Location', 'northeast'); grid on; xlim([0 4000]);

%% Quality Metrics
fprintf('\n=== Quality Metrics ===\n');

noise_orig = std(x(1:min(round(0.5*fs), length(x))));
noise_bp = std(x_bp(1:min(round(0.5*fs), length(x_bp))));
noise_wiener = std(x_wiener(1:min(round(0.5*fs), length(x_wiener))));
noise_ss = std(x_ss(1:min(round(0.5*fs), length(x_ss))));
noise_adaptive = std(x_adaptive(1:min(round(0.5*fs), length(x_adaptive))));
noise_med = std(x_med(1:min(round(0.5*fs), length(x_med))));
noise_final = std(x_final(1:min(round(0.5*fs), length(x_final))));

fprintf('Noise Floor (std dev):\n');
fprintf('  Original:              %.6f\n', noise_orig);
fprintf('  Bandpass:              %.6f (%.1f%% reduction)\n', noise_bp, (1-noise_bp/noise_orig)*100);
fprintf('  Wiener Filtered:       %.6f (%.1f%% reduction)\n', noise_wiener, (1-noise_wiener/noise_orig)*100);
fprintf('  Spectral Subtraction:  %.6f (%.1f%% reduction)\n', noise_ss, (1-noise_ss/noise_orig)*100);
fprintf('  Adaptive Gating:       %.6f (%.1f%% reduction)\n', noise_adaptive, (1-noise_adaptive/noise_orig)*100);
fprintf('  Median Filtered:       %.6f (%.1f%% reduction)\n', noise_med, (1-noise_med/noise_orig)*100);
fprintf('  Combined Best:         %.6f (%.1f%% reduction)\n', noise_final, (1-noise_final/noise_orig)*100);

fprintf('\n=== Files Generated ===\n');
fprintf('Stage 0: stage_0_original.wav\n');
fprintf('Stage 1: stage_1_bandpass.wav\n');
fprintf('Stage 2: Multiple denoising algorithms\n');
fprintf('  - stage_2a_wiener_filtered.wav\n');
fprintf('  - stage_2b_spectral_subtraction.wav\n');
fprintf('  - stage_2c_adaptive_gating.wav (AI-inspired)\n');
fprintf('  - stage_2d_median_filtered.wav\n');
fprintf('Stage 3: Voice amplified versions (2.5x)\n');
fprintf('  - stage_3_wiener_amplified.wav\n');
fprintf('  - stage_3_spectralsub_amplified.wav\n');
fprintf('  - stage_3_adaptive_amplified.wav\n');
fprintf('  - stage_3_median_amplified.wav\n');
fprintf('Stage 4: stage_4_combined_best.wav\n');

fprintf('\n✅ Processing complete! Listen to each version to find best buzz removal.\n');
fprintf('🎯 Recommended: Try stage_2c (Adaptive) or stage_2a (Wiener) or stage_4 (Combined)\n');