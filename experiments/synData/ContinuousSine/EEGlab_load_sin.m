clear;clc;
%% ----------------- EEGLAB PART -----------------
fprintf('Starting EEGLAB...\n');
eeglab;  % opens EEGLAB
outFolder = 'synData\ContinuousSine';
%% Load BIN + JSON
binFile  = fullfile(outFolder, 'ContinuousSine_continuous.bin');
jsonFile = fullfile(outFolder, 'params.json');

fprintf('Loading BIN + JSON...\n');

% Read JSON
txt = fileread(jsonFile);
params_loaded = jsondecode(txt);

% Read binary (2 x nSamples)
fid = fopen(binFile, 'r');
data_loaded = fread(fid, [2 Inf], 'double'); 
fclose(fid);

% Apply parameters from JSON
srate     = params_loaded.srate;
epoch_win = params_loaded.epoch_window_s;

%% Import into EEGLAB
EEG = pop_importdata( ...
    'dataformat','array', ...
    'data','data_loaded', ...
    'srate',srate);

EEG.setname = 'ContinuousSine';
EEG = eeg_checkset(EEG);

%% Add channel names
EEG.chanlocs = struct('labels', {'SINE','TRIGGER'});
EEG = eeg_checkset(EEG);

%% ------------------- FFT of full continuous data (EEGLAB spectopo) -------------------
fprintf('Computing FFT of continuous signal using spectopo()...\n');

% Compute spectrum using EEGLAB
% Returns:
%   S     → power spectrum in dB
%   freqs → frequency vector
[S, freqs] = spectopo(EEG.data, EEG.pnts, EEG.srate, ...
                      'plot', 'off', ...     % no auto-plot
                      'chanlocs', EEG.chanlocs);

% Extract channel 1 (SINE)
S_ch1 = S(1,:);

% Plot
figure('Name','Spectopo – Continuous Signal','Position',[300 300 800 300]);
plot(freqs, S_ch1, 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Spectopo – Continuous Sine Channel');
grid on;

%% Save PSD (dB) to binary (freqs + PSD_dB)
psdFile = fullfile(outFolder, 'EEGlab_psd_dB_continuous.bin');
fid = fopen(psdFile, 'w');
fwrite(fid, [freqs(:) S_ch1(:)]', 'double');
fclose(fid);
fprintf('Saved PSD (dB) data: %s\n', psdFile);


%% ------------------- Event extraction -------------------
trigger_idx = find(data_loaded(2,:) == 1);

EEG.event = [];
for k = 1:length(trigger_idx)
    EEG.event(k).type = 'trigger';
    EEG.event(k).latency = trigger_idx(k);
    EEG.event(k).urevent = k;
end
EEG = eeg_checkset(EEG,'eventconsistency');

%% ------------------- Epoch -------------------
EEG_ep = pop_epoch(EEG, {'trigger'}, epoch_win, 'epochinfo','yes');

%% ------------------- ERP -------------------
ERP = mean(EEG_ep.data, 3); 
time_ms = linspace(epoch_win(1)*1000, epoch_win(2)*1000, EEG_ep.pnts);

%% Save ERP to binary (time + values)
erpFile = fullfile(outFolder, 'EEGlab_erp_average.bin');
fid = fopen(erpFile,'w');
fwrite(fid, [time_ms(:) ERP(1,:).']', 'double'); 
fclose(fid);

fprintf('Saved ERP average: %s\n', erpFile);

%% ------------------- Plotting -------------------
figure('Name','Channel 1 – Single Trials & ERP','Position',[200 200 900 400]); 
hold on;
plot(time_ms, squeeze(EEG_ep.data(1,:,:)), 'Color', [0.7 0.7 0.7]);
plot(time_ms, ERP(1,:), 'b', 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Amplitude (µV)');
title('10 Hz Sine – Single Trials (gray) & ERP (blue)');
grid on;

fprintf('Done.\n');