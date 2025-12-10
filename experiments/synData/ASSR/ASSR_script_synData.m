%% Continuous 40 Hz ASSR Simulation using SEREEGA
% This script simulates ASSR data using SEREEGA and creates scalp EEG,
% topographies, and PSD plots. 

clear; clc;

%% Add required paths
% Ensure the following are downloaded and available in MATLAB path:
% 1. EEGLAB: https://github.com/sccn/eeglab.git
% 2. SEREEGA: https://github.com/SGaMi/sereega
% 3. New York Head model (ICBM-NY): https://www.parralab.org/nyhead/

addpath(genpath(pwd));  
eeglab; % Start EEGLAB 

%% Simulation parameters
srate = 1000;       % Hz
duration = 5;       % seconds
nSamples = srate * duration;

%% Epoch configuration (single continuous epoch)
epochs = struct();
epochs.n = 1;            
epochs.srate = srate;
epochs.length = duration*1000;  
epochs.prestim = 0;
epochs.marker = 'stimulus';      

%% Lead field
lf = lf_generate_fromnyhead('montage', 'S64');

%% Select a source in auditory cortex (right hemisphere)
coords = [40 -30 5];  
source = lf_get_source_nearest(lf, coords);  
plot_source_location(source, lf, 'mode', '3d');

%% Define continuous 40 Hz ASSR using ERSP class
assrFreq = 40;       
assrAmplitude = 2;   

ersp = struct( ...
    'type', 'ersp', ...
    'frequency', assrFreq, ...
    'amplitude', assrAmplitude, ...
    'modulation', 'none');  

ersp = utl_check_class(ersp);

%% Add brown noise
noise = struct('type','noise','color','brown','amplitude',3);
noise = utl_check_class(noise);

%% Combine into one component
c = struct();
c.source = source;
c.signal = {ersp, noise};
c = utl_check_component(c, lf);

%% Simulate scalp EEG data
scalpdata = generate_scalpdata(c, lf, epochs);

%% Create EEGLAB dataset
EEG = utl_create_eeglabdataset(scalpdata, epochs, lf);

%% Select T8 for plotting (right auditory cortex)
chanLabel = 'T8';
chanIdx = find(strcmp({EEG.chanlocs.labels}, chanLabel));
t8Data = squeeze(scalpdata(chanIdx, :, 1));  
timeVec = (0:EEG.pnts-1)/EEG.srate*1000;  % ms

%% Plot continuous signal at T8
figT8 = figure('Name','T8 ASSR Continuous','Position',[100 100 900 400],'Color','w');
plot(timeVec, t8Data, 'b', 'LineWidth', 1.5);
xlabel('Time (ms)'); ylabel('Amplitude (\muV)');
title(['T8 ASSR Continuous Signal (' num2str(assrFreq) ' Hz)']);
grid on;

%% Topography using spectopo at 40 Hz
dataForTopo = squeeze(scalpdata(:, :, 1));  
[S, freqs] = spectopo(dataForTopo, 0, EEG.srate, 'plot','off');

[~, freqIdx] = min(abs(freqs - assrFreq));
ampASSRTopo = S(:, freqIdx);  

%% Save topography as vector PDF
methodType = 'ASSR';
outputFolder = fullfile('synData', methodType);
if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

pdfTopo = fullfile(outputFolder, ['EEGlab_ASSR_' num2str(assrFreq) 'Hz_Topography.pdf']);
saveTopoplotPDF(ampASSRTopo, EEG.chanlocs, ...
                 ['ASSR ' num2str(assrFreq) ' Hz power'], ...
                 pdfTopo, 'Power (\muV^2/Hz)');
disp(['ASSR topography saved as PDF: ', pdfTopo]);

%% PSD at T8 over full frequency range
figure('Name','T8 PSD','Position',[100 100 900 400],'Color','w');
[St8, freqsT8] = spectopo(t8Data, 0, EEG.srate, 'plot','off');

plot(freqsT8, St8, 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Power (\muV^2/Hz)');
title(['PSD at T8']);
grid on;
xlim([0 EEG.srate/2]);

pdfPSD = fullfile(outputFolder, 'EEGlab_T8_PSD_vector.pdf');
exportgraphics(gcf, pdfPSD, 'ContentType', 'vector');
close(gcf);
disp(['T8 PSD saved as vector PDF: ', pdfPSD]);

%% Save scalp data
outputFile = fullfile(outputFolder, [methodType '_scalpdata']); 
binFilename = [outputFile '.bin'];
fid = fopen(binFilename,'w');
if fid == -1
    error('Cannot open file: %s\n%s', binFilename, msg);
end
fwrite(fid, scalpdata, 'double'); fclose(fid);
disp(['Scalp data saved to ', binFilename]);

%% Create JSON parameters
jsonFilename = fullfile(outputFolder, 'params.json');
createParamsJSON_Simulation(methodType, outputFile, [], jsonFilename, epochs);
