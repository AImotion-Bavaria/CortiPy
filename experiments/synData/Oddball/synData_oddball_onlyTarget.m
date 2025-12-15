%% Oddball Paradigm Simulation – Target Only
clear; clc;
addpath(genpath(pwd));
addpath('H:\01_Projekte\12_EEG_Analysis_Tool\eeglab');

eeglab;

%% Epoch configuration
epochs = struct();
epochs.n = 300;          % total epochs (all target now)
epochs.srate = 1000;
epochs.length = 600;     % ms
epochs.prestim = 0;
epochs.marker = 'stimulus';

% All markers are now target
markers = repmat(2, 1, epochs.n);

%% Lead field
lf = lf_generate_fromnyhead('montage', 'S64');

%% Pick a source (parietal area for P300)
source = lf_get_source_nearest(lf, [0 -60 50]);
plot_source_location(source, lf, 'mode', '3d');

%% ERP definitions
% Shared early sensory response (N1–P2)
N1 = struct('peakLatency', 120, 'peakWidth', 30, 'peakAmplitude', -3);
N1 = utl_check_class(N1, 'type', 'erp');

P2 = struct('peakLatency', 180, 'peakWidth', 35, 'peakAmplitude', 4);
P2 = utl_check_class(P2, 'type', 'erp');

% Target P300 only
P300_target = struct('peakLatency', 300, 'peakWidth', 60, 'peakAmplitude', 8);
P300_target = utl_check_class(P300_target, 'type', 'erp');

%% Noise
noise = struct('type','noise','color','brown','amplitude',3);
noise = utl_check_class(noise);

%% Create component for target only
comp_target = struct();
comp_target.source = source;
comp_target.signal = {N1, P2, P300_target, noise};
comp_target = utl_check_component(comp_target, lf);

%% Generate data
scalpdata = generate_scalpdata(comp_target, lf, epochs);  % 64 × 600 × nTarget

%% Create EEGLAB set
EEG = utl_create_eeglabdataset(scalpdata, epochs, lf);

% Store event info
EEG.event = struct('type', num2cell(markers), 'latency', num2cell((1:epochs.n)*(EEG.pnts)));

%% Plot ERP at parietal site (Pz)
chanIdx = find(strcmp({EEG.chanlocs.labels}, 'Pz'));
tgtAvg = mean(EEG.data(chanIdx,:,:), 3);
t = (0:EEG.pnts-1) / EEG.srate * 1000;

figERP = figure('Position',[100 100 800 400],'Color','w');
plot(t, tgtAvg, 'r', 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Amplitude (µV)');
title('Target ERP at Pz (Target Only)');
grid on;

% Save ERP figure
outputFolder = fullfile('synData','Oddball');
if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end
pdfERP = fullfile(outputFolder,'Oddball_ERP_TargetOnly_Pz.pdf');
exportgraphics(figERP, pdfERP, 'ContentType', 'vector');
close(figERP);
disp(['Target-only ERP at Pz saved as PDF: ', pdfERP]);

%% Compute target topography at 300 ms
timeVec = (0:EEG.pnts-1)/EEG.srate*1000;
[~, samp300] = min(abs(timeVec - 300));
ERP_target = mean(EEG.data(:,:, :), 3);  

% Plot topography
targetTitle = 'Target Only Topography at P300';
targetPDF  = fullfile(outputFolder, 'EEGlab_TargetOnly_Topography.pdf');
saveTopoplotPDF(ERP_target(:,samp300), EEG.chanlocs, targetTitle, targetPDF, 'Amplitude (µV)');
disp(['Target-only topography saved as PDF: ', targetPDF]);

%% Save scalp data
binFilename = fullfile(outputFolder, 'Oddball_TargetOnly_scalpdata.bin');
fid = fopen(binFilename, 'w');
if fid == -1
    error('Could not open file for writing: %s', binFilename);
end
fwrite(fid, scalpdata, 'double');
fclose(fid);
disp(['Target-only scalp data saved to ', binFilename]);

%% Save JSON
jsonFilename = fullfile(outputFolder, 'params_TargetOnly.json');
createParamsJSON_Simulation('Oddball', 'Oddball_TargetOnly_scalpdata', [], jsonFilename, epochs);
