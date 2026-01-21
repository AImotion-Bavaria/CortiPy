%% Continuous SSVEP Simulation using SEREEGA
% This script simulates steady-state visual evoked potentials (SSVEPs)
% and generates scalp EEG, topographies, PSDs, and saves data.

clear; clc;

%% Add required paths
% Ensure the following are downloaded and available in MATLAB path:
% 1. EEGLAB: https://github.com/sccn/eeglab.git
% 2. SEREEGA: https://github.com/SGaMi/sereega
% 3. New York Head model (ICBM-NY): https://www.parralab.org/nyhead/
addpath(genpath(pwd));

%% Simulation parameters
srate = 1000;       % Hz
duration = 5;       % seconds
nSamples = srate * duration;

%% Epoch configuration (single continuous epoch)
epochs = struct();
epochs.n = 1;             % one long epoch
epochs.srate = srate;
epochs.length = duration*1000;  % in ms
epochs.prestim = 0;
epochs.marker = 'stimulus';      % required by JSON function

%% Lead field
lf = lf_generate_fromnyhead('montage', 'S64');

%% Select a source in visual cortex (occipital region)
coords = [5 -80 20];  % approximate V1 location
source = lf_get_source_nearest(lf, coords);  
plot_source_location(source, lf, 'mode', '3d');

%% Define continuous SSVEP using ERSP class
ssvepFreq = 10;      % Hz (e.g., 10 Hz flicker)
ssvepAmplitude = 3;  % µV

ersp = struct( ...
    'type', 'ersp', ...
    'frequency', ssvepFreq, ...
    'amplitude', ssvepAmplitude, ...
    'modulation', 'none');  % continuous, unmodulated

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

%% Plot continuous signal at Oz
chanIdx = find(strcmp({EEG.chanlocs.labels}, 'Oz'));
ozData = squeeze(scalpdata(chanIdx, :, :));  
timeVec = (0:EEG.pnts-1)/EEG.srate*1000;  % ms

figure('Name','Oz SSVEP Continuous','Position',[100 100 900 400]);
plot(timeVec, ozData, 'b', 'LineWidth', 1.5);
xlabel('Time (ms)'); ylabel('Amplitude (\muV)');
title('Oz SSVEP Continuous Signal');
grid on;

%% Topography using EEGLAB spectopo
dataForTopo = squeeze(scalpdata(:, :, 1));  % channels x time
[S, freqs] = spectopo(dataForTopo, 0, EEG.srate, 'plot','off');

% Find closest frequency index to SSVEP
[~, freqIdx] = min(abs(freqs - ssvepFreq));
ampSSVEPTopo = S(:, freqIdx);  % amplitude per channel at SSVEP frequency

% Plot scalp map
figure('Visible','off','Color','w');  % offscreen figure with white background
topoplot(ampSSVEPTopo, EEG.chanlocs, ...
    'electrodes','on', ...
    'maplimits','absmax', ...
    'style','fill', ...
    'whitebk','on');
title(['SSVEP ' num2str(ssvepFreq) ' Hz amplitude (spectopo)']);

% Add colorbar
c = colorbar;
c.Label.String = '\muV';
c.Label.FontSize = 12;

% Ensure output folder exists
methodType = 'SSVEP';
outputFolder = fullfile('synData', methodType);
if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

% Save topography as vector PDF
pdfFilename = fullfile(outputFolder, ['EEGlab_SSVEP_' num2str(ssvepFreq) 'Hz_Topography.pdf']);
saveTopoplotPDF(ampSSVEPTopo, EEG.chanlocs, ...
                 ['SSVEP ' num2str(ssvepFreq) ' Hz power'], ...
                 pdfFilename, 'Power (\muV^2/Hz)');
disp(['SSVEP topography saved as PDF: ', pdfFilename]);


%% PSD at Oz over full frequency range
chanIdx = find(strcmp({EEG.chanlocs.labels}, 'Oz'));
ozData = squeeze(scalpdata(chanIdx, :, 1));   % single epoch = time series

figure('Name','Oz PSD','Position',[100 100 900 400],'Color','w');

% Compute PSD using spectopo (no plotting)
[Soz, freqsOZ] = spectopo(ozData, 0, EEG.srate, 'plot','off');

% Plot PSD
plot(freqsOZ, Soz, 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Power (\muV^2/Hz)');
title('PSD at Oz');
grid on;
xlim([0 EEG.srate/2]);
% Save as vector PDF
outputFolder = fullfile('synData','SSVEP');
if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end
pdfFilename = fullfile(outputFolder, 'EEGlab_Oz_PSD_vector.pdf');
exportgraphics(gcf, pdfFilename, 'ContentType', 'vector');
close(gcf);

disp(['Oz PSD saved as vector PDF: ', pdfFilename]);
%% Save scalp data
outputFile = fullfile(outputFolder, [methodType '_scalpdata']); 
binFilename = [outputFile '.bin'];
[fid, msg] = fopen(binFilename,'w');
if fid == -1
    error('Cannot open file: %s\n%s', binFilename, msg);
end
fwrite(fid, scalpdata, 'double'); fclose(fid);
disp(['Scalp data saved to ', binFilename]);

%% Create JSON parameters
jsonFilename = fullfile(outputFolder, 'params.json');
createParamsJSON_Simulation(methodType, outputFile, [], jsonFilename, epochs);
