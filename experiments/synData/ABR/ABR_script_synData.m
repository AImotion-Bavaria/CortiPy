%% Minimal ABR simulation (right auditory cortex, T8) using SEREEGA
% This script simulates auditory brainstem responses (ABRs) and generates
% scalp EEG, topographies, and ERP plots.

clear; clc;

%% Add required paths
% Ensure the following are downloaded and available in MATLAB path:
% 1. EEGLAB: https://github.com/sccn/eeglab.git
% 2. SEREEGA: https://github.com/SGaMi/sereega
% 3. New York Head model (ICBM-NY): https://www.parralab.org/nyhead/
addpath(genpath(pwd));  
eeglab; % Start EEGLAB 

%% Epoch configuration
epochs = struct();
epochs.n = 200;         
epochs.srate = 25000;    
epochs.length = 15;  
epochs.prestim = 0;
epochs.marker = 'stimulus';

%% Lead field
lf = lf_generate_fromnyhead('montage', 'S64');

%% Source (right auditory cortex for consistency with ASSR)
source = lf_get_source_nearest(lf, [40 -30 5]);  
plot_source_location(source, lf, 'mode', '3d');

%% Define smooth, interconnected ABR peaks (nV)
ABR_signal = {};

% --- Wave I (~1.5 ms) ---
ABR_signal{end+1} = utl_check_class(struct('peakLatency',1.5,'peakWidth',1.2,'peakAmplitude', 1.500),'type','erp');  
ABR_signal{end+1} = utl_check_class(struct('peakLatency',1.8,'peakWidth',1.2,'peakAmplitude',-.500),'type','erp'); 

% --- Wave II (~2.8 ms) ---
ABR_signal{end+1} = utl_check_class(struct('peakLatency',2.7,'peakWidth',1.5,'peakAmplitude', 2.000),'type','erp');  
ABR_signal{end+1} = utl_check_class(struct('peakLatency',3.0,'peakWidth',1.5,'peakAmplitude',-1.800),'type','erp'); 

% --- Wave III (~4.2 ms) ---
ABR_signal{end+1} = utl_check_class(struct('peakLatency',4.1,'peakWidth',1.5,'peakAmplitude', 2.500),'type','erp');  
ABR_signal{end+1} = utl_check_class(struct('peakLatency',4.5,'peakWidth',1.5,'peakAmplitude',-1.300),'type','erp'); 

% --- Wave IV (~5.5 ms) ---
ABR_signal{end+1} = utl_check_class(struct('peakLatency',5.4,'peakWidth',1.5,'peakAmplitude', 2.500),'type','erp');  
ABR_signal{end+1} = utl_check_class(struct('peakLatency',5.8,'peakWidth',1.5,'peakAmplitude',-1.200),'type','erp'); 

% --- Wave V (~6.8 ms) ---
ABR_signal{end+1} = utl_check_class(struct('peakLatency',6.7,'peakWidth',3.5,'peakAmplitude', 5.000),'type','erp');  
ABR_signal{end+1} = utl_check_class(struct('peakLatency',7.2,'peakWidth',2.5,'peakAmplitude',-4.500),'type','erp'); 

%% Noise (white)
noise = struct('type','noise','color','brown','amplitude',3);

%% Create component
c = struct();
c.source = source;
c.signal = [ABR_signal, {noise}];
c = utl_check_component(c, lf);

%% Generate scalp data
scalpdata = generate_scalpdata(c, lf, epochs);

%% Create EEGLAB dataset
EEG = utl_create_eeglabdataset(scalpdata, epochs, lf);

%% Compute sample index for Wave V (~7 ms)
nSamples = size(EEG.data, 2);  
sampV = round(7/epochs.length * nSamples);  
ABR_avg = mean(EEG.data, 3);                  

%% Ensure output folder exists
methodType = 'ABR';
outputFolder = fullfile('synData', methodType);
if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

%% Save ABR Topography as vector PDF (Wave V)
pdfTopo = fullfile(outputFolder, 'EEGlab_ABR_WaveV_Topography.pdf');
saveTopoplotPDF(ABR_avg(:,sampV), EEG.chanlocs, ...
                 'ABR Topography at Wave V', ...
                 pdfTopo, '\muV');
disp(['ABR topography saved as PDF: ', pdfTopo]);

%% Plot and save ERP at T8 (same as ASSR)
chanIdx = find(strcmp({EEG.chanlocs.labels}, 'T8'));
erpT8 = squeeze(EEG.data(chanIdx,:,:));
timeVec = (0:EEG.pnts-1)/EEG.srate*1000;  

figERP = figure('Name','ABR at T8','Position',[100 100 900 400],'Color','w');
plot(timeVec, erpT8, 'Color',[0.7 0.7 0.7]); hold on;
plot(timeVec, mean(erpT8,2), 'b', 'LineWidth', 2);
xlabel('Time (ms)'); ylabel('Amplitude (\muV)');
title('ABR Single-Trials (gray) & Average (blue) at T8');
grid on;

pdfERP = fullfile(outputFolder, 'EEGlab_ABR_T8_ERP_vector.pdf');
exportgraphics(figERP, pdfERP, 'ContentType', 'vector');
close(figERP);
disp(['ABR T8 ERP saved as vector PDF: ', pdfERP]);

%% Save scalp data
outputFile = fullfile(outputFolder, [methodType '_scalpdata']); 
binFilename = [outputFile '.bin'];
fid = fopen(binFilename, 'w');
if fid == -1
    error('Cannot open file: %s', binFilename);
end
fwrite(fid, scalpdata, 'double');  
fclose(fid);
disp(['Scalp data saved to ', binFilename]);

%% Create JSON
jsonFilename = fullfile(outputFolder, 'params.json');
createParamsJSON_Simulation('ABR', 'ABR_scalpdata', [], jsonFilename, epochs);
