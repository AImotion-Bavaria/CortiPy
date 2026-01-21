%% Continuous sine + trigger -> save -> load in EEGLAB -> epoch -> average
clear; clc;

%% Add EEGLAB
addpath('H:\01_Projekte\12_EEG_Analysis_Tool\eeglab');

%% ----------------- PARAMETERS -----------------
srate      = 1000;      % Hz
duration_s = 120;       % total recording length in seconds
freq       = 10;        % sine frequency Hz
amp_uV     = 1;      % amplitude in microvolt (µV)
trigger_interval_s = 1; % trigger every 1 second
epoch_win = [-0.2 0.8]; % epoch window in seconds
outFolder = fullfile('synData','ContinuousSine');
if ~exist(outFolder,'dir'), mkdir(outFolder); end
%% ------------------------------------------------

%% Create time vector and channels
nSamples = duration_s * srate;
t = (0:nSamples-1) / srate;

% Channel 1: continuous sine
ch1 = amp_uV * sin(2*pi*freq*t);

% Channel 2: trigger (0 everywhere, 1 every 1s)
ch2 = zeros(1, nSamples);
trigger_samples = round(1 : trigger_interval_s*srate : nSamples);
ch2(trigger_samples) = 1;

% Combine data [channels x samples]
data = [ch1; ch2];

%% Save to binary
binFile = fullfile(outFolder, 'ContinuousSine_continuous.bin');
fid = fopen(binFile, 'w');
fwrite(fid, data, 'double');
fclose(fid);
fprintf('Saved binary: %s\n', binFile);

%% Save JSON
params = struct();
params.method = 'ContinuousSine';
params.srate = srate;
params.duration_s = duration_s;
params.freq_hz = freq;
params.amplitude_uV = amp_uV;
params.trigger_interval_s = trigger_interval_s;
params.epoch_window_s = epoch_win;

jsonFile = fullfile(outFolder, 'params.json');
fid = fopen(jsonFile,'w');
fwrite(fid, jsonencode(params), 'char');
fclose(fid);
fprintf('Saved JSON: %s\n', jsonFile);

%% ----------------- PLOTS -----------------
figure('Name','Synthetic Continuous Sine');

subplot(3,1,1);
plot(t(1:3000), ch1(1:3000));
xlabel('Time (s)');
ylabel('Amplitude (µV)');
title('Sine Wave (first 3 seconds)');

subplot(3,1,2);
stem(t(trigger_samples(1:10)), ones(1,10), 'filled');
xlabel('Time (s)');
ylabel('Trigger');
title('First 10 Triggers');
xlim([0 10]);

subplot(3,1,3);
plot(t, ch2);
xlabel('Time (s)');
ylabel('Trigger');
title('Trigger Train (full duration)');
ylim([-0.1 1.1]);
