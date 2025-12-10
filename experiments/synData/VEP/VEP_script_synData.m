%% Continuous VEP Simulation using SEREEGA
% This script simulates visual evoked potentials (VEPs), generates
% scalp EEG, ERP plots, topographies, and saves data.

clear; clc;

%% Add required paths
% Ensure the following are downloaded and added to MATLAB path:
% 1. EEGLAB: https://github.com/sccn/eeglab.git
% 2. SEREEGA: https://github.com/SGaMi/sereega
% 3. New York Head model (ICBM-NY): https://www.parralab.org/nyhead/
addpath(genpath(pwd));
eeglab;

%% Epoch configuration

epochs = struct();
epochs.n = 100;        % number of epochs
epochs.srate = 1000;   % sampling rate in Hz
epochs.length = 500;   % epoch length in ms
epochs.prestim = 0;  % 100 ms pre-stimulus
epochs.marker = 'stimulus';
srate = epochs.srate;       % e.g., 1000 Hz
prestim = epochs.prestim;   % e.g., 100 ms

%% Obtain lead field (64-channel montage)
lf = lf_generate_fromnyhead('montage', 'S64');

%% Select a source in the right visual cortex (approximate coordinates)
source = lf_get_source_nearest(lf, [5 -80 20]); % MNI coords
plot_source_location(source, lf, 'mode', '3d');

%% Define ERP (VEP) activation with N75-P100-N135
% N75
erp1 = struct('peakLatency', 75, 'peakWidth', 20, 'peakAmplitude', -5, ...
              'peakLatencyDv', 5, 'peakAmplitudeDv', 1);
erp1 = utl_check_class(erp1, 'type', 'erp');

% P100
erp2 = struct('peakLatency', 100, 'peakWidth', 45, 'peakAmplitude', 10, ...
              'peakLatencyDv', 5, 'peakAmplitudeDv', 2);
erp2 = utl_check_class(erp2, 'type', 'erp');

% N135
erp3 = struct('peakLatency', 135, 'peakWidth', 40, 'peakAmplitude', -7, ...
              'peakLatencyDv', 5, 'peakAmplitudeDv', 1);
erp3 = utl_check_class(erp3, 'type', 'erp');

%% Define noise (a bit stronger)
noise = struct('type','noise','color','brown','amplitude',3);
noise = utl_check_class(noise);

%% Combine ERP peaks and noise into a component
c = struct();
c.source = source;
c.signal = {erp1, erp2, erp3, noise};
c = utl_check_component(c, lf);


%% Simulate scalp EEG data
scalpdata = generate_scalpdata(c, lf, epochs);

%% Create EEGLAB dataset
EEG = utl_create_eeglabdataset(scalpdata, epochs, lf);

%% Optional: add ICA weights for ground truth
%EEG = utl_add_icaweights_toeeglabdataset(EEG, c, lf);



%% Extract and plot single-trial time signal at Oz
chanIdx = find(strcmpi({EEG.chanlocs.labels}, 'Oz'));
if isempty(chanIdx)
    error('Channel "Oz" not found in EEG.chanlocs.');
end
ozData = squeeze(EEG.data(chanIdx, :, :));  % [time x trials]
timeVec = (0:EEG.pnts-1)/EEG.srate*1000 - epochs.prestim;  % ms

% Plot ERP
figERP = figure('Name','Oz VEP Single-Trial + Average','Position',[100 100 900 400]);
plot(timeVec, ozData, 'Color',[0.7 0.7 0.7]); hold on;       % individual trials
plot(timeVec, mean(ozData,2), 'b', 'LineWidth', 2);          % average
xlabel('Time (ms)'); ylabel('Amplitude (µV)');
title('Oz VEP: Single-Trials (gray) & Average (blue)');
grid on;

% Save ERP as vector PDF
% Ensure output folder exists
methodType = 'VEP';
outputFolder = fullfile('synData', methodType);
if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

pdfERP = fullfile(outputFolder, 'EEGlab_Oz_VEP_vector.pdf');
exportgraphics(figERP, pdfERP, 'ContentType', 'vector');
close(figERP);
disp(['Oz VEP ERP saved as vector PDF: ', pdfERP]);

%% Plot VEP topographies at selected time points
timePoints = [100];  % ms
VEP_avg = mean(EEG.data, 3);        % average across trials

for t = 1:length(timePoints)
    sampIdx = round(timePoints(t)/1000 * EEG.srate);  
    
    pdfTopo = fullfile(outputFolder, ['EEGlab_VEP_Topography_' num2str(timePoints(t)) 'ms.pdf']);
u
    disp(['VEP topography saved as PDF: ', pdfTopo]);
end



%% File name for binary data
outputFile = fullfile(outputFolder, [methodType '_scalpdata']); % without extension
binFilename = [outputFile '.bin'];

%% Save scalp data to binary
fid = fopen(binFilename, 'w');
if fid == -1
    error('Could not open file for writing: %s', binFilename);
end
fwrite(fid, scalpdata, 'double');  % use 'single' if you want smaller files
fclose(fid);

disp(['Scalp data saved to ', binFilename]);

% Create JSON
jsonFilename = fullfile(outputFolder, 'params.json');
createParamsJSON_Simulation('VEP', 'VEP_scalpdata', [], jsonFilename, epochs);