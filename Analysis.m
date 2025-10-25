%% Advanced Speech Enhancement with Adaptive Amplitude Restoration
% Tests multiple algorithms with dynamic amplitude restoration
clear; clc; close all;

%% Load and prepare audio
[x, fs] = audioread('2m.m4a');
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

%% STAGE 2C: Adaptive Spectral Gating 
fprintf('\n=== Stage 2C: Adaptive Spectral Gating ===\n');

frame_size = round(0.020 * fs);
hop_size = round(0.010 * fs);
win = hann(frame_size, 'periodic');
nfft = 2^nextpow2(frame_size * 2);

% Learn noise profile starting from 0.03s to avoid initial spikes
noise_start_time = 0.03; % Start at 30ms to skip initial artifacts
noise_start_sample = round(noise_start_time * fs);

% Learn from longer duration for 2-min audio: use 5 seconds or 5% of audio
noise_learning_duration = min(5.0, length(x_bp)/fs * 0.05);
noise_learning_samples = round(noise_learning_duration * fs);
noise_end_sample = min(noise_start_sample + noise_learning_samples, length(x_bp));

noise_frames = floor((noise_end_sample - noise_start_sample - frame_size) / hop_size);

fprintf('  Noise learning: %.2fs to %.2fs (%.2f seconds)\n', ...
        noise_start_time, noise_end_sample/fs, noise_learning_duration);

% Estimate noise statistics with variance
noise_mean = zeros(nfft, 1);
noise_var = zeros(nfft, 1);

for i = 1:noise_frames
    idx = noise_start_sample + (i-1) * hop_size;
    if idx + frame_size - 1 > length(x_bp), break; end
    
    frame = x_bp(idx:idx+frame_size-1) .* win;
    frame_mag = abs(fft(frame, nfft));
    
    noise_mean = noise_mean + frame_mag;
    noise_var = noise_var + frame_mag.^2;
end

noise_mean = noise_mean / noise_frames;
noise_var = (noise_var / noise_frames) - noise_mean.^2;
noise_std = sqrt(max(noise_var, eps));

% Process with adaptive threshold
num_frames = floor((length(x_bp) - frame_size) / hop_size) + 1;
x_adaptive = zeros(length(x_bp), 1);
window_sum = zeros(length(x_bp), 1);

% Adaptive parameters
sensitivity = 2.5;
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
    
    % Adaptive gate
    snr_estimate = (frame_mag - noise_mean) ./ (noise_std + eps);
    
    % Soft gating function
    gate = 1 ./ (1 + exp(-sensitivity * (snr_estimate - 1)));
    
    % Temporal smoothing 
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

% Smart amplitude restoration using threshold-based approach
% Calculate noise threshold from first 0.5s (starting at 0.03s to avoid spikes)
threshold_start = round(0.03 * fs);
threshold_end = min(round(0.5 * fs), length(x_adaptive));
noise_section = x_adaptive(threshold_start:threshold_end);
noise_threshold = max(abs(noise_section)) * 1.5; % 1.5x max of noise section

fprintf('  Noise threshold: %.4f\n', noise_threshold);

% Find signal peak using 99.5th percentile (to ignore outlier spikes)
sorted_abs = sort(abs(x_adaptive));
percentile_idx = round(0.995 * length(sorted_abs));
signal_peak_99p5 = sorted_abs(percentile_idx);
absolute_peak = max(abs(x_adaptive));

fprintf('  99.5th percentile peak: %.4f, Absolute peak: %.4f\n', signal_peak_99p5, absolute_peak);

% Calculate gain to bring 99.5th percentile to 0.98
target_peak = 0.98;
calculated_gain = target_peak / signal_peak_99p5;

% Limit gain to prevent over-amplification
max_gain = 15.0;
calculated_gain = min(calculated_gain, max_gain);

fprintf('  Calculated gain: %.2fx\n', calculated_gain);

% Apply smart amplification
x_adaptive_amplified = zeros(size(x_adaptive));

for i = 1:length(x_adaptive)
    sample_abs = abs(x_adaptive(i));
    
    if sample_abs < noise_threshold
        % Below threshold: apply full gain
        gain = calculated_gain;
    else
        % Above threshold: blend between calculated gain and minimal gain
        excess_ratio = (sample_abs - noise_threshold) / (signal_peak_99p5 - noise_threshold + eps);
        excess_ratio = min(excess_ratio, 1.0);
        
        % Interpolate between calculated_gain and 1.0
        gain = calculated_gain * (1 - 0.7 * excess_ratio) + 1.0 * (0.7 * excess_ratio);
    end
    
    x_adaptive_amplified(i) = x_adaptive(i) * gain;
end

% Clip any remaining outlier spikes (above 0.99)
x_adaptive_amplified(x_adaptive_amplified > 0.99) = 0.99;
x_adaptive_amplified(x_adaptive_amplified < -0.99) = -0.99;

x_adaptive = x_adaptive_amplified;

fprintf('  Final peak: %.4f, Final RMS: %.4f (%.1f%% of original)\n', ...
        max(abs(x_adaptive)), rms(x_adaptive), (rms(x_adaptive)/original_rms)*100);

audiowrite('stage_2c_adaptive_gating.wav', x_adaptive, fs);
fprintf('✓ Adaptive spectral gating complete\n');

%% STAGE 3: Adaptive Amplitude Restoration
fprintf('\n=== Stage 3: Adaptive Amplitude Restoration ===\n');

denoised_versions = {x_wiener, x_ss, x_adaptive};
version_names = {'Wiener', 'SpectralSub', 'Adaptive'};

amplified_versions = cell(3, 1);

for v = 1:3
    signal = denoised_versions{v};
    
    fprintf('\nProcessing %s:\n', version_names{v});
    
    % Voice Activity Detection
    frame_size = round(0.025 * fs);
    hop_size = round(0.010 * fs);
    num_frames = floor((length(signal) - frame_size) / hop_size) + 1;
    voice_mask = zeros(length(signal), 1);
    
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
    
    % Calculate RMS of voice regions in original vs denoised
    voice_indices = voice_mask > 0.5;
    
    if sum(voice_indices) > 0
        % Get corresponding regions from original signal
        x_aligned = x;
        if length(x_aligned) > length(signal)
            x_aligned = x_aligned(1:length(signal));
        elseif length(x_aligned) < length(signal)
            signal = signal(1:length(x_aligned));
            voice_indices = voice_indices(1:length(x_aligned));
        end
        
        original_voice_rms = rms(x_aligned(voice_indices));
        denoised_voice_rms = rms(signal(voice_indices));
        
        % Calculate required gain to restore original amplitude
        if denoised_voice_rms > 0
            restoration_gain = original_voice_rms / denoised_voice_rms;
        else
            restoration_gain = 1.0;
        end
        
        % Limit gain to reasonable range (prevent over-amplification)
        restoration_gain = min(restoration_gain, 8.0);
        restoration_gain = max(restoration_gain, 1.0);
        
        fprintf('  Voice RMS - Original: %.4f, Denoised: %.4f\n', ...
                original_voice_rms, denoised_voice_rms);
        fprintf('  Calculated restoration gain: %.2fx\n', restoration_gain);
    else
        % Fallback if no voice detected
        restoration_gain = 2.5;
        fprintf('  No voice detected, using default gain: %.2fx\n', restoration_gain);
    end
    
    % Apply adaptive amplification only to voice regions
    amplified = signal .* (1 + voice_mask * (restoration_gain - 1));
    
    % Percentile-based normalization to handle spikes
    sorted_abs = sort(abs(amplified));
    percentile_idx = round(0.995 * length(sorted_abs));
    peak_99p5 = sorted_abs(percentile_idx);
    absolute_peak = max(abs(amplified));
    
    fprintf('  99.5th percentile: %.4f, Absolute peak: %.4f\n', peak_99p5, absolute_peak);
    
    % Normalize to 99.5th percentile
    amplified = amplified / peak_99p5 * 0.99;
    
    % Clip any remaining outliers
    amplified(amplified > 0.99) = 0.99;
    amplified(amplified < -0.99) = -0.99;
    
    final_rms = rms(amplified);
    if sum(voice_indices) > 0
        final_voice_rms = rms(amplified(voice_indices));
        fprintf('  Final voice RMS: %.4f (%.1f%% of original)\n', ...
                final_voice_rms, (final_voice_rms/original_voice_rms)*100);
    end
    fprintf('  Final total RMS: %.4f (%.1f%% of original)\n', ...
            final_rms, (final_rms/original_rms)*100);
    fprintf('  Final peak amplitude: %.4f\n', max(abs(amplified)));
    
    amplified_versions{v} = amplified;
    
    filename = sprintf('stage_3_%s_amplified.wav', lower(version_names{v}));
    audiowrite(filename, amplified, fs);
    fprintf('✓ %s restored (%.2fx adaptive gain)\n', version_names{v}, restoration_gain);
end

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

% Percentile-based normalization to handle spikes
sorted_abs = sort(abs(x_final));
percentile_idx = round(0.995 * length(sorted_abs));
peak_99p5 = sorted_abs(percentile_idx);
absolute_peak = max(abs(x_final));

fprintf('Combined - 99.5th percentile: %.4f, Absolute peak: %.4f\n', peak_99p5, absolute_peak);

% Normalize to 99.5th percentile
x_final = x_final / peak_99p5 * 0.99;

% Clip any remaining outliers
x_final(x_final > 0.99) = 0.99;
x_final(x_final < -0.99) = -0.99;

fprintf('Final combined RMS: %.4f (%.1f%% of original)\n', rms(x_final), (rms(x_final)/original_rms)*100);
fprintf('Final combined peak: %.4f\n', max(abs(x_final)));

audiowrite('stage_4_combined.wav', x_final, fs);
fprintf('✓ Combined enhancement complete\n');

%% Visualization
fprintf('\n=== Generating Visualizations ===\n');

figure('Position', [50 50 1600 1200]);

signals = {x, x_bp, x_wiener, x_ss, x_adaptive, amplified_versions{3}, x_final};
titles = {'Original', 'Bandpass', 'Wiener Filtered', 'Spectral Sub', ...
          'Adaptive Gating', 'Adaptive + Restored', 'Combined Best'};

for i = 1:7
    sig = signals{i};
    t = (0:length(sig)-1) / fs;
    
    % Waveform
    subplot(7, 2, 2*i-1);
    plot(t, sig, 'LineWidth', 0.5);
    title(titles{i}, 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on; axis tight; ylim([-1 1]);
    
    % Add RMS indicator
    sig_rms = rms(sig);
    text(0.02, 0.9, sprintf('RMS: %.4f', sig_rms), ...
         'Units', 'normalized', 'FontSize', 8, 'BackgroundColor', 'white');
    
    % Spectrogram
    subplot(7, 2, 2*i);
    spectrogram(sig, hann(512), 256, 1024, fs, 'yaxis');
    title([titles{i} ' - Spectrogram']);
    colormap jet; caxis([-80 0]);
end

sgtitle('Adaptive Amplitude Restoration Speech Enhancement', 'FontSize', 14, 'FontWeight', 'bold');

%% Quality Metrics
fprintf('\n=== Quality Metrics ===\n');

noise_orig = std(x(1:min(round(0.5*fs), length(x))));
noise_bp = std(x_bp(1:min(round(0.5*fs), length(x_bp))));
noise_wiener = std(x_wiener(1:min(round(0.5*fs), length(x_wiener))));
noise_ss = std(x_ss(1:min(round(0.5*fs), length(x_ss))));
noise_adaptive = std(x_adaptive(1:min(round(0.5*fs), length(x_adaptive))));
noise_final = std(x_final(1:min(round(0.5*fs), length(x_final))));

fprintf('Noise Floor (std dev):\n');
fprintf('  Original:              %.6f\n', noise_orig);
fprintf('  Bandpass:              %.6f (%.1f%% reduction)\n', noise_bp, (1-noise_bp/noise_orig)*100);
fprintf('  Wiener Filtered:       %.6f (%.1f%% reduction)\n', noise_wiener, (1-noise_wiener/noise_orig)*100);
fprintf('  Spectral Subtraction:  %.6f (%.1f%% reduction)\n', noise_ss, (1-noise_ss/noise_orig)*100);
fprintf('  Adaptive Gating:       %.6f (%.1f%% reduction)\n', noise_adaptive, (1-noise_adaptive/noise_orig)*100);
fprintf('  Combined Best:         %.6f (%.1f%% reduction)\n', noise_final, (1-noise_final/noise_orig)*100);

fprintf('\n=== Amplitude Metrics ===\n');
fprintf('RMS Levels:\n');
fprintf('  Original:              %.6f\n', original_rms);
fprintf('  Wiener (restored):     %.6f (%.1f%% of original)\n', rms(amplified_versions{1}), rms(amplified_versions{1})/original_rms*100);
fprintf('  SpectralSub (restored):%.6f (%.1f%% of original)\n', rms(amplified_versions{2}), rms(amplified_versions{2})/original_rms*100);
fprintf('  Adaptive (restored):   %.6f (%.1f%% of original)\n', rms(amplified_versions{3}), rms(amplified_versions{3})/original_rms*100);
fprintf('  Combined Best:         %.6f (%.1f%% of original)\n', rms(x_final), rms(x_final)/original_rms*100);

fprintf('\n=== Files Generated ===\n');
fprintf('Stage 0: stage_0_original.wav\n');
fprintf('Stage 1: stage_1_bandpass.wav\n');
fprintf('Stage 2: Multiple denoising algorithms\n');
fprintf('  - stage_2a_wiener_filtered.wav\n');
fprintf('  - stage_2b_spectral_subtraction.wav\n');
fprintf('  - stage_2c_adaptive_gating.wav\n');
fprintf('Stage 3: Adaptively restored versions\n');
fprintf('  - stage_3_wiener_amplified.wav\n');
fprintf('  - stage_3_spectralsub_amplified.wav\n');
fprintf('  - stage_3_adaptive_amplified.wav\n');
fprintf('Stage 4: stage_4_combined.wav\n');

fprintf('\n✓ Processing complete! Adaptive restoration maintains original amplitude.\n');
