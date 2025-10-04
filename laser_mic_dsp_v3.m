%% Laser Microphone DSP Comparison - Enhanced Version
% Compare FIR-only, Wiener, Spectral Subtraction, Enhanced Spectral, and combinations

clear; clc; close all;

%% PARAMETERS
fs = 48000;           % Sampling frequency (Hz)
nBits = 16;           % Bit depth for recording(USB 7.1 external sound cards)
nChannels = 1;        % Mono
recordDuration = 10;   % Record length (seconds  

[x,fs0] = audioread('raw_recording2.wav');
fprintf('file fs = %d Hz\n', fs0); 
spectrogram(x, hann(2048), 1024, 4096, fs0,'yaxis');

% DSP params
hpCutoff = 100;        % High-pass cutoff for DC/rumble removal
bpLow = 100;          % Bandpass low cutoff
bpHigh = 18000;       % Bandpass high cutoff
Nfir = 480;           % FIR filter length (# of taps) 

% y = filter(bang, 1, x);

% % Or for zero-phase filtering (better in DSP)
% y = filtfilt(b, 1, x);

% % Visualize filter response
% fvtool(bang, 1)          

% for h = [10,30,60,100]
%     [b,a] = butter(4, h/(fs/2), 'high');
%     y = filtfilt(b,a,x);
%     soundsc(y,fs); pause(1+length(y)/fs);
% end

% transitions = [100,200,400]; % Hz
% Ntests = [128,256,512];
% results = [];
% for t = transitions
%   for N = Ntests
%     N_est = round(fs / t);
%     N_use = max(32, min(N, N_est)); % keep reasonable
%     b_fir = firpm(N_use, [0 (bpLow-20)/(fs/2) bpLow/(fs/2) bpHigh/(fs/2) (bpHigh+200)/(fs/2) 1], [0 0 1 1 0 0]);
%     y = filtfilt(b_fir,1,x);
%     s = snr(y); % crude
%     results = [results; t N_use s];
%   end
% end
% disp(array2table(results, 'VariableNames', {'TransitionHz','Nfir','SNRdB'}));
% Noise reduction params
alpha = 1.2;          % Spectral subtraction over-subtraction factor
beta = 0.02;          % Spectral subtraction floor (musical noise suppression)
k_gate = 2.0;         % Gating threshold multiplier         

%% SECTION 1: RECORD RAW AUDIO
% recObj = audiorecorder(fs, nBits, nChannels);
% disp('Recording raw signal...');
% recordblocking(recObj, recordDuration);
% rawSignal = getaudiodata(recObj);
% audiowrite('raw_recording.wav', rawSignal, fs);
% disp('Saved raw_recording.wav');

%% SECTION 2: RUN ALL VARIANTS
fprintf('Processing audio variants...\n');

[x_fir,fs] = preprocess_fir('raw_recording2.wav',hpCutoff,bpLow,bpHigh,Nfir);
fprintf('  FIR preprocessing complete\n');

x_spec = spectral_subtract(x_fir,fs,alpha,beta,k_gate);
fprintf('  Spectral subtraction complete\n');

x_wiener = wiener_filter(x_fir,fs);
fprintf('  Wiener filtering complete\n');

x_combo = spectral_subtract(x_wiener,fs,alpha,beta,k_gate);
fprintf('  Wiener + Spectral combo complete\n');


% Save all variants
audiowrite('proc_fir.wav',x_fir,fs);
audiowrite('proc_spec.wav',x_spec,fs);
audiowrite('proc_wiener.wav',x_wiener,fs);
audiowrite('proc_combo.wav',x_combo,fs);

fprintf('All variants saved\n\n');

%% SECTION 3: COMPARISONS
[raw,~] = audioread('raw_recording2.wav');

L = length(raw);
t = (0:L-1)/fs;

x_fir             = align_and_normalize(x_fir, L);
x_spec            = align_and_normalize(x_spec, L);
x_wiener          = align_and_normalize(x_wiener, L);
x_combo           = align_and_normalize(x_combo, L);

% Waveform comparison
figure('Name', 'Waveform Comparison', 'Position', [100 100 1200 900]);
subplot(8,1,1); plot(t,raw); title('Raw'); xlim([0 max(t)]); ylabel('Amplitude');
subplot(8,1,2); plot(t,x_fir); title('FIR-only'); xlim([0 max(t)]); ylabel('Amplitude');
subplot(8,1,3); plot(t,x_spec); title('Spectral Subtraction'); xlim([0 max(t)]); ylabel('Amplitude');
subplot(8,1,5); plot(t,x_wiener); title('Wiener Filter'); xlim([0 max(t)]); ylabel('Amplitude');
subplot(8,1,7); plot(t,x_combo); title('Wiener + Spectral'); xlim([0 max(t)]); ylabel('Amplitude');

% Spectrogram comparison - Original methods
figure('Name', 'Spectrogram Comparison - Original', 'Position', [150 150 1200 800]);
subplot(2,3,1); spectrogram(raw,hamming(1024),512,1024,fs,'yaxis'); title('Raw'); colorbar;
subplot(2,3,2); spectrogram(x_fir,hamming(1024),512,1024,fs,'yaxis'); title('FIR-only'); colorbar;
subplot(2,3,3); spectrogram(x_spec,hamming(1024),512,1024,fs,'yaxis'); title('Spectral Subtraction'); colorbar;
subplot(2,3,4); spectrogram(x_wiener,hamming(1024),512,1024,fs,'yaxis'); title('Wiener'); colorbar;
subplot(2,3,5); spectrogram(x_combo,hamming(1024),512,1024,fs,'yaxis'); title('Wiener + Spectral'); colorbar;

%% SECTION 4: SNR METRICS
[ref,~] = audioread('Not _Like_Us.wav'); 
ref = align_and_normalize(ref, L);

fprintf('=== SNR COMPARISON ===\n');
%snr_value = snr(clean, noisy-clean);
snr_raw            = snr(raw);
snr_fir            = snr(x_fir);
snr_spec           = snr(x_spec);
snr_wiener         = snr(x_wiener);
snr_combo          = snr(x_combo);
% 
% % PESQ (Audio Toolbox: narrowband 'nb' or wideband 'wb')
% pesq_raw    = pesq(ref, raw, fs);
% pesq_fir    = pesq(ref, x_fir, fs);
% pesq_spec   = pesq(ref, x_spec, fs);
% pesq_wiener = pesq(ref, x_wiener, fs);
% pesq_combo  = pesq(ref, x_combo, fs);

% % STOI (from File Exchange implementation)
% stoi_raw    = stoi(ref, raw, fs);
% stoi_fir    = stoi(ref, x_fir, fs);
% stoi_spec   = stoi(ref, x_spec, fs);
% stoi_wiener = stoi(ref, x_wiener, fs);
% stoi_combo  = stoi(ref, x_combo, fs);

%Problem: This assumes the input has both clean + noise separated, but here we don t! So SNR here is “relative” unless you compare against a known clean reference.

fprintf('SNR Raw:                    %.2f dB\n', snr_raw);
fprintf('SNR FIR:                    %.2f dB\n', snr_fir);
fprintf('SNR Spectral Subtraction:   %.2f dB\n', snr_spec);
fprintf('SNR Wiener:                 %.2f dB\n', snr_wiener);
fprintf('SNR Combo (W+S):            %.2f dB\n', snr_combo);

% Bar chart comparison
figure('Name', 'SNR Comparison', 'Position', [250 250 1000 500]);
snr_values = [snr_raw, snr_fir, snr_spec, ...
              snr_wiener, snr_combo];
labels = {'Raw', 'FIR', 'Spectral','Wiener', 'Combo'};
bar(snr_values);
set(gca, 'XTickLabel', labels);
xtickangle(45);
ylabel('SNR (dB)');
title('SNR Comparison of All Methods');
grid on;

% % Print results
% fprintf('SNR Raw:        %.2f dB | PESQ: %.2f | STOI: %.2f\n', snr_raw, pesq_raw, stoi_raw);
% fprintf('SNR FIR:        %.2f dB | PESQ: %.2f | STOI: %.2f\n', snr_fir, pesq_fir, stoi_fir);
% fprintf('SNR Spectral:   %.2f dB | PESQ: %.2f | STOI: %.2f\n', snr_spec, pesq_spec, stoi_spec);
% fprintf('SNR Wiener:     %.2f dB | PESQ: %.2f | STOI: %.2f\n', snr_wiener, pesq_wiener, stoi_wiener);
% fprintf('SNR Combo:      %.2f dB | PESQ: %.2f | STOI: %.2f\n', snr_combo, pesq_combo, stoi_combo);
% 
% % Optional: plot comparison
% figure('Name','Quality Metrics','Position',[300 300 1000 500]);
% bar([snr_raw snr_fir snr_spec snr_wiener snr_combo; ...
%      pesq_raw pesq_fir pesq_spec pesq_wiener pesq_combo; ...
%      stoi_raw stoi_fir stoi_spec stoi_wiener stoi_combo]');
% legend('SNR (dB)','PESQ','STOI');
% set(gca,'XTickLabel',{'Raw','FIR','Spectral','Wiener','Combo'});
% xtickangle(45);
% ylabel('Metric Value');
% title('Speech Enhancement Metrics');
% grid on;

%% SECTION 5: DSP FUNCTIONS

function [x_out,fs] = preprocess_fir(infile,hpCutoff,bpLow,bpHigh,Nfir)
    % Enhanced FIR preprocessing with pre-emphasis
    [x,fs] = audioread(infile);
    if size(x,2)>1, x = mean(x,2); end

    % High-pass DC removal (higher order for better rejection)
    [b,a] = butter(4, hpCutoff/(fs/2), 'high'); 
    x = filtfilt(b,a,x);

    % FIR bandpass with improved transition bands
    f = [0 (bpLow-20)/(fs/2) bpLow/(fs/2) ...
         bpHigh/(fs/2) (bpHigh+200)/(fs/2) 1];
    a_spec = [0 0 1 1 0 0];
    b_fir = firpm(Nfir, f, a_spec);
    x_out = real(filtfilt(b_fir,1,x));
    
    % Normalize to prevent clipping
    x_out = x_out / max(abs(x_out)+eps) * 0.95;
end

function x_out = spectral_subtract2(x,fs,alpha,beta,k_gate,noiseFile)
    % Spectral subtraction using a separate noise-only file
    
    % === Step 1: Load noise reference ===
    [noiseSig, fs_noise] = audioread(noiseFile);
    if fs_noise ~= fs
        error('Noise file must have the same sampling rate as input signal');
    end
    if size(noiseSig,2) > 1
        noiseSig = mean(noiseSig,2); % mono
    end

    % === Step 2: STFT parameters ===
    winLen = round(0.032*fs);
    hop = round(winLen/2);
    win = hamming(winLen,'periodic');
    nfft = 2^nextpow2(winLen);

    % === Step 3: Noise spectrum estimate from noise file ===
    [S_noise,~,~] = stft(noiseSig, fs, 'Window', win, ...
        'OverlapLength', winLen-hop, 'FFTLength', nfft);
    noiseSpec = mean(abs(S_noise),2);  % average across noise frames

    % === Step 4: STFT of speech signal ===
    [S,~,~] = stft(x, fs, 'Window', win, ...
        'OverlapLength', winLen-hop, 'FFTLength', nfft);
    mag = abs(S); 
    phase = angle(S);
    mag_clean = zeros(size(mag));

    % === Step 5: Spectral subtraction frame-by-frame ===
    nFrames = size(S,2);
    for t = 1:nFrames
        mag_t = mag(:,t);

        % Subtract noise estimate
        mag_sub = mag_t - alpha*noiseSpec;
        mag_sub(mag_sub < 0) = 0;

        % Apply spectral floor
        mag_clean(:,t) = max(mag_sub, beta*mag_t);

        % Spectral gating
        thresh = k_gate*noiseSpec;
        gain = mag_sub >= thresh;
        mag_clean(:,t) = mag_clean(:,t) .* gain;
    end

    % === Step 6: Reconstruct ===
    S_proc = mag_clean .* exp(1j*phase);
    x_out = istft(S_proc, fs, 'Window', win, ...
        'OverlapLength', winLen-hop, 'FFTLength', nfft);

    % === Step 7: Align and normalize ===
    L = min(length(x), length(x_out));
    x_out = real(x_out(1:L));                 % remove imaginary residue
    x_out = x_out / max(abs(x_out)+eps) * 0.98; % normalize to avoid clipping
end

function x_out = spectral_subtract_enhanced(x,fs,alpha,beta,k_gate)
    % Enhanced spectral subtraction with multiple improvements
    
    %% STFT Setup
    winLen = round(0.032*fs);
    hop = round(winLen/2);
    win = hamming(winLen,'periodic');
    nfft = 2^nextpow2(winLen);
    
    % Pre-emphasis for better speech processing
    x_preemph = filter([1 -0.97], 1, x);
    
    [S, f, ~] = stft(x_preemph, fs, 'Window', win, ...
        'OverlapLength', winLen-hop, 'FFTLength', nfft);
    
    [nFreq, nFrames] = size(S);
    
    %% Improved Noise Estimation - Minimum Statistics
    mag = abs(S);
    winSize = max(10, round(1.0 * fs / hop));
    noiseSpec = zeros(nFreq, 1);
    
    for freq = 1:nFreq
        noiseSpec(freq) = min(movmin(mag(freq, 1:min(100, nFrames)), ...
                              min(winSize, nFrames)));
    end
    
    % Smooth noise estimate across frequency
    noiseSpec = smoothdata(noiseSpec, 'gaussian', 7);
    
    %% Process each frame
    phase = angle(S);
    mag_clean = zeros(size(mag));
    mag_prev = mag(:, 1);
    
    for t = 1:nFrames
        mag_t = mag(:, t);
        
        %% Improved VAD
        energyRatio = sum(mag_t.^2) / (sum(noiseSpec.^2) + eps);
        spectralFlatness = geomean(mag_t) / (mean(mag_t) + eps);
        
        % Voice frequency band (300-3400 Hz)
        voiceBand = (f >= 300) & (f <= 3400);
        voiceEnergy = sum(mag_t(voiceBand).^2);
        totalEnergy = sum(mag_t.^2) + eps;
        voiceRatio = voiceEnergy / totalEnergy;
        
        isSpeech = (energyRatio > 2.5) && ...
                   (spectralFlatness < 0.3) && ...
                   (voiceRatio > 0.35);
        
        % Update noise during non-speech
        if ~isSpeech
            noiseSpec = 0.95*noiseSpec + 0.05*mag_t;
        end
        
        %% Adaptive parameters
        localSNR = 10*log10(sum(mag_t.^2) / (sum(noiseSpec.^2) + eps));
        
        if localSNR < 0
            alpha_adapt = alpha * 1.5;
            beta_adapt = beta * 0.5;
        elseif localSNR < 10
            alpha_adapt = alpha;
            beta_adapt = beta;
        else
            alpha_adapt = alpha * 0.8;
            beta_adapt = beta * 2.0;
        end
        
        %% Spectral Subtraction
        mag_sub = mag_t - alpha_adapt * noiseSpec;
        mag_sub(mag_sub < 0) = 0;
        mag_clean(:, t) = max(mag_sub, beta_adapt * mag_t);
        
        %% Spectral gate with smooth transitions
        thresh = k_gate * noiseSpec;
        gain = double(mag_sub >= thresh);
        gain = smoothdata(gain, 'gaussian', 3);
        mag_clean(:, t) = mag_clean(:, t) .* gain;
        
        %% Musical noise reduction
        if t > 1
            alpha_smooth = 0.6;
            onset = mag_t > 2 * mag_prev;
            mag_clean(~onset, t) = alpha_smooth * mag_clean(~onset, t-1) + ...
                                   (1 - alpha_smooth) * mag_clean(~onset, t);
        end
        
        mag_prev = mag_t;
    end
    
    %% Temporal smoothing
    for freq = 1:nFreq
        mag_clean(freq, :) = smoothdata(mag_clean(freq, :), 'movmean', 3);
    end
    
    %% Reconstruct signal
    S_proc = mag_clean .* exp(1j * phase);
    x_out = istft(S_proc, fs, 'Window', win, ...
        'OverlapLength', winLen-hop, 'FFTLength', nfft);
    
    % De-emphasis
    x_out = filter(1, [1 -0.97], x_out);
    
    %% Post-processing
    L = min(length(x), length(x_out));
    x_out = real(x_out(1:L));
    
    % High-pass to remove residual rumble
    [b, a] = butter(2, 80/(fs/2), 'high');
    x_out = filtfilt(b, a, x_out);
    
    % Intelligent normalization
    peak = max(abs(x_out));
    if peak > 0
        target = 0.95 * max(abs(x));
        x_out = x_out * (target / peak);
    end
end

function x_out = spectral_subtract(x,fs,alpha,beta,k_gate)
    % Original spectral subtraction
    winLen = round(0.032*fs);
    hop = round(winLen/2);
    win = hamming(winLen,'periodic');
    nfft = 2^nextpow2(winLen);
    [S,~,~] = stft(x, fs, 'Window', win, ...
        'OverlapLength', winLen-hop, 'FFTLength', nfft);

    nFrames = size(S,2);
    nNoise = max(3, round(0.10 * nFrames));
    noiseSpec = mean(abs(S(:,1:nNoise)),2);

    mag = abs(S); phase = angle(S);
    mag_clean = zeros(size(mag));

    for t = 1:nFrames
        mag_t = mag(:,t);
        if mean(mag_t) < 1.5*mean(noiseSpec)
            noiseSpec = 0.95*noiseSpec + 0.05*mag_t;
        end

        mag_sub = mag_t - alpha*noiseSpec;
        mag_sub(mag_sub<0) = 0;
        mag_clean(:,t) = max(mag_sub, beta*mag_t);

        thresh = k_gate*noiseSpec;
        gain = mag_sub >= thresh;
        mag_clean(:,t) = mag_clean(:,t).*gain;
    end

    S_proc = mag_clean .* exp(1j*phase);
    x_out = istft(S_proc, fs, 'Window', win, ...
        'OverlapLength', winLen-hop, 'FFTLength', nfft);
    L = min(length(x), length(x_out));
    x_out = real(x_out(1:L));
    x_out = x_out / max(abs(x_out)+eps) * 0.98;
end


function x_out = wiener_filter(x,fs)
    % Original Wiener filter
    winLen = round(0.032*fs);
    hop = round(winLen/2);
    win = hamming(winLen,'periodic');
    nfft = 2^nextpow2(winLen);
    [S,~,~] = stft(x, fs, 'Window', win, ...
        'OverlapLength', winLen-hop, 'FFTLength', nfft);

    nFrames = size(S,2);
    nNoise = max(3, round(0.10 * nFrames));
    noiseSpec = mean(abs(S(:,1:nNoise)),2);

    mag = abs(S); phase = angle(S);
    mag_clean = zeros(size(mag));

    for t = 1:nFrames
        mag_t = mag(:,t);
        if mean(mag_t) < 1.5*mean(noiseSpec)
            noiseSpec = 0.95*noiseSpec + 0.05*mag_t;
        end

        noisePow = noiseSpec.^2;
        gainW = max((mag_t.^2 - noisePow) ./ (mag_t.^2 + eps), 0);
        mag_clean(:,t) = gainW .* mag_t;
    end

    S_proc = mag_clean .* exp(1j*phase);
    x_out = istft(S_proc, fs, 'Window', win, ...
        'OverlapLength', winLen-hop, 'FFTLength', nfft);
    L = min(length(x), length(x_out));
    x_out = real(x_out(1:L));
    x_out = x_out / max(abs(x_out)+eps) * 0.98;
end

function y = align_and_normalize(y, L)
    % Force real
    y = real(y);

    % Match length
    if length(y) < L
        y(end+1:L) = 0;
    else
        y = y(1:L);
    end

    % Normalize to prevent clipping
    if max(abs(y)) > 0
        y = y / max(abs(y)) * 0.98;
    end
end