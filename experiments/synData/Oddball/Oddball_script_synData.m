%% Oddball Paradigm Simulation using SEREEGA
% Simulates standard vs. target ERPs and generates scalp EEG, ERP plots,
% topographies, and saves data in vector PDFs.

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
epochs.n = 300;          % total epochs
epochs.ratio = 0.8;      % 80% standard, 20% target
epochs.srate = 1000;
epochs.length = 600;     % ms
epochs.prestim = 0;
epochs.marker = 'stimulus';

% Condition markers
nStand = round(epochs.n * epochs.ratio);
nTarget = epochs.n - nStand;
markers = [repmat(1,1,nStand), repmat(2,1,nTarget)];
markers = markers(randperm(length(markers)));  % randomize sequence

%% Lead field
lf = lf_generate_fromnyhead('montage', 'S64');

%% Pick a source (parietal area for P300)
source = lf_get_source_nearest(lf, [0 -60 50]);  % slightly more posterior and dorsal

plot_source_location(source, lf, 'mode', '3d');

%% ERP definitions
% Shared early sensory response (N1–P2)
N1 = struct('peakLatency', 120, 'peakWidth', 30, 'peakAmplitude', -3);
N1 = utl_check_class(N1, 'type', 'erp');

P2 = struct('peakLatency', 180, 'peakWidth', 35, 'peakAmplitude', 4);
P2 = utl_check_class(P2, 'type', 'erp');

% Standard has weak/no P300
P300_standard = struct('peakLatency', 300, 'peakWidth', 60, 'peakAmplitude', 1);
P300_standard = utl_check_class(P300_standard, 'type', 'erp');

% Target has stronger P300
P300_target = struct('peakLatency', 300, 'peakWidth', 60, 'peakAmplitude', 8);
P300_target = utl_check_class(P300_target, 'type', 'erp');

%% Noise
noise = struct('type','noise','color','brown','amplitude',3);
noise = utl_check_class(noise);

%% Create components for standard + target
comp_standard = struct();
comp_standard.source = source;
comp_standard.signal = {N1, P2, P300_standard, noise};
comp_standard = utl_check_component(comp_standard, lf);

comp_target = struct();
comp_target.source = source;
comp_target.signal = {N1, P2, P300_target, noise};
comp_target = utl_check_component(comp_target, lf);

%% Generate data trial-by-trial
scalpdata = zeros(length(lf.chanlocs), epochs.length * epochs.srate / 1000, epochs.n);

%% Generate standard and target sets separately (full batches)
epochs_standard = epochs;
epochs_standard.n = nStand;

epochs_target = epochs;
epochs_target.n = nTarget;

scalp_standard = generate_scalpdata(comp_standard, lf, epochs_standard);  % 64 × 600 × nStand
scalp_target   = generate_scalpdata(comp_target,   lf, epochs_target);    % 64 × 600 × nTarget

%% Allocate full dataset
scalpdata = zeros(size(scalp_standard,1), size(scalp_standard,2), epochs.n);

%% Insert trials into final array according to marker order
stdCounter = 1;
tgtCounter = 1;

for i = 1:epochs.n
    if markers(i) == 1
        scalpdata(:,:,i) = scalp_standard(:,:,stdCounter);
        stdCounter = stdCounter + 1;
    else
        scalpdata(:,:,i) = scalp_target(:,:,tgtCounter);
        tgtCounter = tgtCounter + 1;
    end
end


%% Create EEGLAB set
EEG = utl_create_eeglabdataset(scalpdata, epochs, lf);

% Store condition labels
EEG.event = struct('type', num2cell(markers), 'latency', num2cell((1:epochs.n)*(EEG.pnts)));

%% Plot ERP for parietal (Pz)
chanIdx = find(strcmp({EEG.chanlocs.labels}, 'Pz'));

stdIdx = find(markers == 1);
tgtIdx = find(markers == 2);

stdAvg = mean(EEG.data(chanIdx,:,stdIdx), 3);
tgtAvg = mean(EEG.data(chanIdx,:,tgtIdx), 3);

t = (0:EEG.pnts-1) / EEG.srate * 1000;

figERP = figure('Position',[100 100 800 400],'Color','w');
plot(t, stdAvg, 'b', 'LineWidth', 2); hold on;
plot(t, tgtAvg, 'r', 'LineWidth', 2);
legend({'Standard','Target'});
xlabel('Time (ms)');
ylabel('Amplitude (µV)');
title('Oddball ERP at Pz');
grid on;

% Save ERP figure as vector PDF
outputFolder = fullfile('synData','Oddball');
if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end
pdfERP = fullfile(outputFolder,'Oddball_ERP_Pz.pdf');
exportgraphics(figERP, pdfERP, 'ContentType', 'vector');
close(figERP);

disp(['Oddball ERP at Pz saved as vector PDF: ', pdfERP]);


%% Compute condition-specific averages for topography
ERP_standard = mean(EEG.data(:,:,stdIdx), 3);  % chan x time
ERP_target   = mean(EEG.data(:,:,tgtIdx), 3);  % chan x time

% Find sample closest to 300 ms
timeVec = (0:EEG.pnts-1)/EEG.srate*1000;
[~, samp300] = min(abs(timeVec - 300));

%% Plot non-target (standard) topography
figure('Name','Standard (Non-target) Topography','Position',[100 100 600 500]);
topoplot(ERP_standard(:,samp300), EEG.chanlocs, 'electrodes','on', 'maplimits','absmax');
colorbar; % add color scale
clim('auto'); % automatic scaling for clarity
title('Standard (Non-target) Topography at 300 ms');

%% Plot target topography
figure('Name','Target Topography','Position',[100 100 600 500]);
topoplot(ERP_target(:,samp300), EEG.chanlocs, 'electrodes','on', 'maplimits','absmax');
colorbar; % add color scale
clim('auto'); % automatic scaling for clarity
title('Target Topography at 300 ms');
% Choose method
methodType = 'Oddball';  % or 'ABR'
% Create output folder if it does not exist
outputFolder = fullfile('synData', methodType);
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Save topoplot as PDF

%% Plot target topography and save as vector PDF
targetTitle = 'Oddball Topography at P300';
targetPDF  = fullfile(outputFolder, 'EEGlab_Oddball_Target_Topography.pdf');
saveTopoplotPDF(ERP_target(:,samp300), EEG.chanlocs, targetTitle, targetPDF, 'µV');
disp(['Target topography saved as PDF: ', targetPDF]);

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
createParamsJSON_Simulation('Odball', 'Oddball_scalpdata', [], jsonFilename, epochs);