%% Speech Enhancement Metrics Analysis
% Analyzes processed audio files and computes SNR, PESQ, and STOI metrics
% Includes multiple SNR calculation methods for comparison

clear; clc; close all;

%% Configuration
fprintf('=== SPEECH ENHANCEMENT METRICS ANALYSIS ===\n\n');

% Define all processed files to analyze
audio_files = {
    'stage_0_original.wav'
    'stage_1_bandpass.wav'
    'stage_2a_wiener_filtered.wav'
    'stage_2b_spectral_subtraction.wav'
    'stage_2c_adaptive_gating.wav'
    'stage_3_wiener_amplified.wav'
    'stage_3_spectralsub_amplified.wav'
    'stage_3_adaptive_amplified.wav'
    'stage_4_combined.wav'
};

stage_names = {
    'Original'
    'Bandpass Filter'
    'Wiener Filter'
    'Spectral Subtraction'
    'Adaptive Gating'
    'Wiener + Amplified'
    'SpectralSub + Amplified'
    'Adaptive + Amplified'
    'Combined'
};

%% Load Audio Files
fprintf('Loading audio files...\n');
signals = cell(length(audio_files), 1);
fs = 0;
valid_files = true(length(audio_files), 1);

for i = 1:length(audio_files)
    if exist(audio_files{i}, 'file')
        [sig, fs_temp] = audioread(audio_files{i});
        
        % Convert to mono if needed
        if size(sig, 2) > 1
            sig = mean(sig, 2);
        end
        
        signals{i} = sig;
        if fs == 0
            fs = fs_temp;
        end
        fprintf('  ✓ Loaded: %s\n', audio_files{i});
    else
        fprintf('  ✗ Missing: %s\n', audio_files{i});
        valid_files(i) = false;
    end
end

% Filter out missing files
signals = signals(valid_files);
stage_names = stage_names(valid_files);
audio_files = audio_files(valid_files);

if isempty(signals)
    error('No audio files found! Please run the enhancement script first.');
end

fprintf('\nSample rate: %d Hz\n', fs);
fprintf('Files loaded: %d/%d\n\n', sum(valid_files), length(valid_files));

%% Load Clean Reference Signal (if available)
has_reference = false;
clean_ref = [];

% Try multiple possible clean reference filenames
reference_files = {'Fences_mono.wav', 'clean_reference.wav', 'reference.wav'};

for i = 1:length(reference_files)
    if exist(reference_files{i}, 'file')
        fprintf('Loading clean reference: %s\n', reference_files{i});
        [clean_ref, fs_clean] = audioread(reference_files{i});
        
        % Resample if needed
        if fs_clean ~= fs
            fprintf('  Resampling reference from %d Hz to %d Hz\n', fs_clean, fs);
            clean_ref = resample(clean_ref, fs, fs_clean);
        end
        
        % Convert to mono
        if size(clean_ref, 2) > 1
            clean_ref = mean(clean_ref, 2);
        end
        
        % Normalize
        clean_ref = clean_ref / max(abs(clean_ref));
        
        has_reference = true;
        fprintf('  ✓ Clean reference loaded successfully\n\n');
        break;
    end
end

if ~has_reference
    fprintf('⚠ No clean reference signal found.\n');
    fprintf('  Searched for: %s\n', strjoin(reference_files, ', '));
    fprintf('  SNR will be estimated using alternative methods.\n\n');
end

%% Initialize Metrics Storage
n_signals = length(signals);

% SNR variants
snr_with_reference = zeros(n_signals, 1);
snr_estimated = zeros(n_signals, 1);
snr_manual = zeros(n_signals, 1);

% PESQ and STOI
pesq_values = zeros(n_signals, 1);
stoi_values = zeros(n_signals, 1);

% Additional metrics
noise_floor = zeros(n_signals, 1);
signal_power = zeros(n_signals, 1);

%% ========== SNR CALCULATION ==========
%% Method 1: SNR with Clean Reference (SEGMENTED APPROACH)
if has_reference
    fprintf('--- Method 1: SNR Using Clean Reference (Segmented) ---\n');
    
    for i = 1:n_signals
        sig = signals{i};
        
        % Match lengths
        min_len = min(length(sig), length(clean_ref));
        sig_trim = sig(1:min_len);
        ref_trim = clean_ref(1:min_len);
        
        % Voice Activity Detection on reference
        frame_size = round(0.025 * fs);
        hop_size = round(0.010 * fs);
        num_frames = floor((length(ref_trim) - frame_size) / hop_size) + 1;
        
        voice_mask = zeros(length(ref_trim), 1);
        
        for j = 1:num_frames
            start_idx = (j-1) * hop_size + 1;
            end_idx = min(start_idx + frame_size - 1, length(ref_trim));
            
            frame = ref_trim(start_idx:end_idx);
            energy = sum(frame.^2) / length(frame);
            zcr = sum(abs(diff(sign(frame)))) / (2 * length(frame));
            
            % Voice detection
            if energy > 0.001 && zcr > 0.03 && zcr < 0.25
                voice_mask(start_idx:end_idx) = 1;
            end
        end
        
        % Identify speech and silence regions
        speech_indices = voice_mask > 0.5;
        silence_indices = voice_mask <= 0.5;
        
        if sum(speech_indices) > 0 && sum(silence_indices) > 0
            % Gain compensation using speech regions only
            speech_sig = sig_trim(speech_indices);
            speech_ref = ref_trim(speech_indices);
            
            sig_rms_speech = rms(speech_sig);
            ref_rms_speech = rms(speech_ref);
            
            if sig_rms_speech > eps && ref_rms_speech > eps
                gain_compensation = ref_rms_speech / sig_rms_speech;
                sig_normalized = sig_trim * gain_compensation;
                
                % Calculate signal power from speech regions in reference
                signal_pwr = sum(ref_trim(speech_indices).^2);
                
                % Calculate noise power from silence regions in processed signal
                % (where there should be no speech, only noise)
                noise_pwr = sum(sig_normalized(silence_indices).^2);
                
                if noise_pwr > eps
                    snr_with_reference(i) = 10 * log10(signal_pwr / noise_pwr);
                else
                    snr_with_reference(i) = Inf;
                end
                
                fprintf('  %s: %.2f dB (Speech: %.1f%%, Gain: %.2fx)\n', ...
                    stage_names{i}, snr_with_reference(i), ...
                    sum(speech_indices)/length(speech_indices)*100, gain_compensation);
            else
                snr_with_reference(i) = NaN;
                fprintf('  %s: N/A (insufficient speech signal)\n', stage_names{i});
            end
        else
            snr_with_reference(i) = NaN;
            fprintf('  %s: N/A (cannot separate speech/silence)\n', stage_names{i});
        end
    end
    fprintf('\n');
else
    fprintf('--- Method 1: Skipped (no reference signal) ---\n\n');
    snr_with_reference(:) = NaN;
end
%% Method 2: Estimated SNR (Noise Floor Analysis)
fprintf('--- Method 2: Estimated SNR (Noise Floor Method) ---\n');

% Use first 0.5 seconds as noise estimate
noise_duration = 0.5; % seconds

for i = 1:n_signals
    sig = signals{i};
    
    % Estimate noise from initial silence
    noise_samples = min(round(noise_duration * fs), length(sig));
    noise_segment = sig(1:noise_samples);
    
    % Estimate signal from remainder
    if length(sig) > noise_samples
        signal_segment = sig(noise_samples+1:end);
    else
        signal_segment = sig;
    end
    
    % Calculate powers
    noise_pwr = mean(noise_segment.^2);
    sig_pwr = mean(signal_segment.^2);
    
    if noise_pwr > eps
        snr_estimated(i) = 10 * log10(sig_pwr / noise_pwr);
    else
        snr_estimated(i) = Inf;
    end
    
    % Store for later analysis
    noise_floor(i) = sqrt(noise_pwr);
    signal_power(i) = sqrt(sig_pwr);
    
    fprintf('  %s: %.2f dB (Noise: %.4f, Signal: %.4f)\n', ...
        stage_names{i}, snr_estimated(i), noise_floor(i), signal_power(i));
end
fprintf('\n');

%% Method 3: Manual SNR (Active Speech vs Background)
fprintf('--- Method 3: Manual SNR (Voice Activity Detection) ---\n');

for i = 1:n_signals
    sig = signals{i};
    
    % Voice Activity Detection (VAD)
    frame_size = round(0.025 * fs); % 25ms frames
    hop_size = round(0.010 * fs);   % 10ms hop
    
    num_frames = floor((length(sig) - frame_size) / hop_size) + 1;
    frame_energy = zeros(num_frames, 1);
    frame_zcr = zeros(num_frames, 1);
    
    for j = 1:num_frames
        start_idx = (j-1) * hop_size + 1;
        end_idx = min(start_idx + frame_size - 1, length(sig));
        
        frame = sig(start_idx:end_idx);
        
        % Energy
        frame_energy(j) = sum(frame.^2) / length(frame);
        
        % Zero-crossing rate
        frame_zcr(j) = sum(abs(diff(sign(frame)))) / (2 * length(frame));
    end
    
    % Adaptive threshold for voice activity
    energy_threshold = median(frame_energy) * 2;
    zcr_low = 0.03;
    zcr_high = 0.30;
    
    % Detect voice frames
    voice_frames = (frame_energy > energy_threshold) & ...
                   (frame_zcr > zcr_low) & ...
                   (frame_zcr < zcr_high);
    
    % Calculate SNR
    if sum(voice_frames) > 0 && sum(~voice_frames) > 0
        voice_power = mean(frame_energy(voice_frames));
        noise_power = mean(frame_energy(~voice_frames));
        
        if noise_power > eps
            snr_manual(i) = 10 * log10(voice_power / noise_power);
        else
            snr_manual(i) = Inf;
        end
        
        voice_pct = sum(voice_frames) / length(voice_frames) * 100;
        fprintf('  %s: %.2f dB (Voice: %.1f%%)\n', ...
            stage_names{i}, snr_manual(i), voice_pct);
    else
        snr_manual(i) = NaN;
        fprintf('  %s: Unable to detect voice/noise separation\n', stage_names{i});
    end
end
fprintf('\n');

%% ========== STOI CALCULATION ==========
fprintf('=== SHORT-TIME OBJECTIVE INTELLIGIBILITY (STOI) ===\n');

has_stoi = false;

if has_reference
    % Try to use STOI function if available
    if exist('stoi', 'file') == 2
        fprintf('Computing STOI scores...\n');
        
        for i = 1:n_signals
            sig = signals{i};
            
            % Match lengths
            min_len = min(length(sig), length(clean_ref));
            sig_trim = sig(1:min_len);
            ref_trim = clean_ref(1:min_len);
            
            try
                stoi_values(i) = stoi(ref_trim, sig_trim, fs);
                fprintf('  %s: %.4f\n', stage_names{i}, stoi_values(i));
                has_stoi = true;
            catch ME
                fprintf('  %s: Error - %s\n', stage_names{i}, ME.message);
                stoi_values(i) = NaN;
            end
        end
    else
        % Use correlation-based approximation
        fprintf('Computing STOI approximation (correlation-based)...\n');
        
        for i = 1:n_signals
            sig = signals{i};
            
            % Match lengths
            min_len = min(length(sig), length(clean_ref));
            sig_trim = sig(1:min_len);
            ref_trim = clean_ref(1:min_len);
            
            if std(sig_trim) > eps && std(ref_trim) > eps
                corr_matrix = corrcoef(ref_trim, sig_trim);
                stoi_values(i) = abs(corr_matrix(1, 2));
                fprintf('  %s: %.4f (approximation)\n', stage_names{i}, stoi_values(i));
                has_stoi = true;
            else
                stoi_values(i) = 0;
                fprintf('  %s: 0.0000 (insufficient variation)\n', stage_names{i});
            end
        end
    end
else
    fprintf('⚠ STOI requires clean reference signal\n');
    stoi_values(:) = NaN;
end
fprintf('\n');

%% ========== SUMMARY TABLE ==========
fprintf('=== COMPREHENSIVE METRICS SUMMARY ===\n\n');

fprintf('%-25s | %12s | %12s | %12s | %10s\n', ...
    'Stage', 'SNR (Ref)', 'SNR (Est)', 'SNR (VAD)', 'STOI');
fprintf('%s\n', repmat('-', 1, 100));

for i = 1:n_signals
    fprintf('%-25s | %12s | %12s | %12s | %10s\n', ...
        stage_names{i}, ...
        format_metric(snr_with_reference(i)), ...
        format_metric(snr_estimated(i)), ...
        format_metric(snr_manual(i)), ...
        format_metric(stoi_values(i)));
end

fprintf('\n');

%% ========== EXPORT RESULTS ==========
fprintf('Exporting results...\n');

% Create results table
results_table = table(...
    stage_names, ...
    snr_with_reference, ...
    snr_estimated, ...
    snr_manual, ...
    stoi_values, ...
    noise_floor, ...
    signal_power, ...
    'VariableNames', {...
        'Stage', ...
        'SNR_Reference_dB', ...
        'SNR_Estimated_dB', ...
        'SNR_VAD_dB', ...
        'STOI', ...
        'Noise_Floor', ...
        'Signal_Power'});

writetable(results_table, 'analysis_results.csv');
fprintf('  ✓ Saved: analysis_results.csv\n');

%% ========== VISUALIZATIONS ==========
fprintf('Generating visualizations...\n');

% Figure 1: SNR Comparison (3 methods)
figure('Name', 'SNR Comparison', 'Position', [100 100 1400 500]);

subplot(1, 3, 1);
if has_reference
    bar(snr_with_reference);
    ylabel('SNR (dB)');
    title('SNR with Clean Reference', 'FontWeight', 'bold');
    set(gca, 'XTickLabel', stage_names, 'XTickLabelRotation', 45);
    grid on;
else
    text(0.5, 0.5, 'No Reference Available', ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
    axis off;
end

subplot(1, 3, 2);
bar(snr_estimated);
ylabel('SNR (dB)');
title('SNR - Noise Floor Estimation', 'FontWeight', 'bold');
set(gca, 'XTickLabel', stage_names, 'XTickLabelRotation', 45);
grid on;

subplot(1, 3, 3);
bar(snr_manual);
ylabel('SNR (dB)');
title('SNR - Voice Activity Detection', 'FontWeight', 'bold');
set(gca, 'XTickLabel', stage_names, 'XTickLabelRotation', 45);
grid on;

sgtitle('Signal-to-Noise Ratio: Method Comparison', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 2: All Metrics
figure('Name', 'Quality Metrics', 'Position', [150 150 1400 500]);

if has_reference
    subplot(1, 3, 1);
    bar(snr_with_reference);
    ylabel('SNR (dB)');
    title('Signal-to-Noise Ratio', 'FontWeight', 'bold');
    set(gca, 'XTickLabel', stage_names, 'XTickLabelRotation', 45);
    grid on;
    
    
    if has_stoi
        subplot(1, 3, 3);
        bar(stoi_values);
        ylabel('STOI Score');
        title('STOI (0-1 scale)', 'FontWeight', 'bold');
        set(gca, 'XTickLabel', stage_names, 'XTickLabelRotation', 45);
        grid on;
        ylim([0 1]);
    end
    
    sgtitle('Speech Quality Metrics', 'FontSize', 14, 'FontWeight', 'bold');
end

% Figure 3: Noise Reduction Analysis
figure('Name', 'Noise Analysis', 'Position', [200 200 1200 500]);

subplot(1, 2, 1);
bar(noise_floor);
ylabel('Amplitude');
title('Noise Floor Analysis', 'FontWeight', 'bold');
set(gca, 'XTickLabel', stage_names, 'XTickLabelRotation', 45);
grid on;

subplot(1, 2, 2);
noise_reduction = (1 - noise_floor / noise_floor(1)) * 100;
bar(noise_reduction);
ylabel('Reduction (%)');
title('Noise Reduction vs Original', 'FontWeight', 'bold');
set(gca, 'XTickLabel', stage_names, 'XTickLabelRotation', 45);
grid on;

sgtitle('Noise Floor and Reduction Analysis', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('  ✓ Visualizations generated\n\n');

%% ========== RECOMMENDATIONS ==========
fprintf('=== RECOMMENDATIONS ===\n\n');

% Find best performing stage for each metric
if has_reference
    [~, best_snr_idx] = max(snr_with_reference);
    fprintf('Best SNR (Reference): %s (%.2f dB)\n', ...
        stage_names{best_snr_idx}, snr_with_reference(best_snr_idx));
end

[~, best_snr_est_idx] = max(snr_estimated);
fprintf('Best SNR (Estimated): %s (%.2f dB)\n', ...
    stage_names{best_snr_est_idx}, snr_estimated(best_snr_est_idx));

[~, best_snr_vad_idx] = max(snr_manual);
fprintf('Best SNR (VAD): %s (%.2f dB)\n', ...
    stage_names{best_snr_vad_idx}, snr_manual(best_snr_vad_idx));

if has_stoi
    [~, best_stoi_idx] = max(stoi_values);
    fprintf('Best STOI: %s (%.4f)\n', ...
        stage_names{best_stoi_idx}, stoi_values(best_stoi_idx));
end

fprintf('\n=== ANALYSIS COMPLETE ===\n');

%% Helper Function
function str = format_metric(value)
    if isnan(value)
        str = 'N/A';
    elseif isinf(value)
        str = 'Inf';
    else
        str = sprintf('%.3f', value);
    end
end