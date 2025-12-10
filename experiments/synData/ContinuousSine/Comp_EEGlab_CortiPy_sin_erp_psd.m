%% ============================================================
%      COMPARE EEGLAB vs CORTIPY PSD + ERP
% ============================================================
clear; clc;

% ---- Paths to binary files ----
inFolder = 'synData/ContinuousSine';

% EEGLAB files
EEG_psd_file = fullfile(inFolder, 'EEGlab_psd_dB_continuous.bin');
EEG_erp_file = fullfile(inFolder, 'EEGlab_erp_average.bin');

% CortiPy files
CP_psd_file  = fullfile(inFolder, 'Cortipy_psd_dB_continuous.bin');
CP_erp_file  = fullfile(inFolder, 'Cortipy_erp_average.bin');

%% ------------------ Load EEGLAB PSD ------------------
fid = fopen(EEG_psd_file,'r');
EEG_psd = fread(fid, [2 Inf], 'double');
fclose(fid);

freqs_EEG   = EEG_psd(1,:);
psd_EEG_dB  = EEG_psd(2,:);

%% ------------------ Load CortiPy PSD ------------------
fid = fopen(CP_psd_file,'r');
CP_psd = fread(fid, [2 Inf], 'double');
fclose(fid);

freqs_CP   = CP_psd(1,:);
psd_CP_dB  = CP_psd(2,:);

%% ------------------ Load EEGLAB ERP ------------------
fid = fopen(EEG_erp_file,'r');
EEG_erp = fread(fid, [2 Inf], 'double');
fclose(fid);

time_EEG = EEG_erp(1,:);
erp_EEG  = EEG_erp(2,:);

%% ------------------ Load CortiPy ERP ------------------
fid = fopen(CP_erp_file,'r');
CP_erp = fread(fid, [2 Inf], 'double');
fclose(fid);

time_CP = CP_erp(1,:);
erp_CP  = CP_erp(2,:);

%% ============================================================
%                    CORRELATION
% ============================================================

% Align lengths (in case CortiPy or EEGLAB padded differently)
min_psd_len = min(length(psd_EEG_dB), length(psd_CP_dB));
min_erp_len = min(length(erp_EEG),    length(erp_CP));

% Crop
psd_EEG_crop = psd_EEG_dB(1:min_psd_len);
psd_CP_crop  = psd_CP_dB(1:min_psd_len);

erp_EEG_crop = erp_EEG(1:min_erp_len);
erp_CP_crop  = erp_CP(1:min_erp_len);

% Pearson correlation
corr_psd = corr(psd_EEG_crop(:), psd_CP_crop(:));
corr_erp = corr(erp_EEG_crop(:), erp_CP_crop(:));

fprintf('\n================ CORRELATION RESULTS ================\n');
fprintf('PSD  correlation (EEGLAB vs CortiPy): %.4f\n', corr_psd);
fprintf('ERP  correlation (EEGLAB vs CortiPy): %.4f\n', corr_erp);
fprintf('=====================================================\n\n');

%% ============================================================
%                    PLOTTING
% ============================================================

figure('Name','PSD Comparison','Position',[100 200 900 350]);
plot(freqs_EEG(1:min_psd_len), psd_EEG_crop, 'b', 'LineWidth', 1.5); hold on;
plot(freqs_CP(1:min_psd_len),  psd_CP_crop, 'r--', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title(sprintf('PSD Comparison (corr = %.3f)', corr_psd));
legend('EEGLAB','CortiPy');
grid on;

figure('Name','ERP Comparison','Position',[100 600 900 350]);
plot(time_EEG(1:min_erp_len), erp_EEG_crop, 'b', 'LineWidth', 1.5); hold on;
plot(time_CP(1:min_erp_len), erp_CP_crop, 'r--', 'LineWidth', 1.5);
xlabel('Time (ms)');
ylabel('Amplitude (µV)');
title(sprintf('ERP Comparison (corr = %.3f)', corr_erp));
legend('EEGLAB','CortiPy');
grid on;
