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
%                    NUMERICAL SIMILARITY
% ============================================================

% PSD similarity
rmse_psd = sqrt(mean((psd_EEG_dB - psd_CP_dB).^2));
mae_psd  = mean(abs(psd_EEG_dB - psd_CP_dB));

% ERP similarity
rmse_erp = sqrt(mean((erp_EEG - erp_CP).^2));
mae_erp  = mean(abs(erp_EEG - erp_CP));

fprintf('\n============== NUMERICAL SIMILARITY RESULTS =============\n');
fprintf('PSD  RMSE (EEGLAB vs CortiPy): %.4f dB\n', rmse_psd);
fprintf('PSD  MAE (EEGLAB vs CortiPy): %.4f dB\n', mae_psd);
fprintf('ERP  RMSE (EEGLAB vs CortiPy): %.4f µV\n', rmse_erp);
fprintf('ERP  MAE (EEGLAB vs CortiPy): %.4f µV\n', mae_erp);
fprintf('==========================================================\n\n');

%% ============================================================
%                     PLOTTING
% ============================================================

figure('Name','PSD Comparison','Position',[100 200 900 350]);
plot(freqs_EEG, psd_EEG_dB, 'b', 'LineWidth', 1.5); hold on;
plot(freqs_CP,  psd_CP_dB, 'r', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title(sprintf('PSD Comparison (RMSE = %.3f dB, MAE = %.3f dB)', rmse_psd, mae_psd));
legend('EEGLAB','CortiPy');
grid on;

figure('Name','ERP Comparison','Position',[100 600 900 350]);
plot(time_EEG, erp_EEG, 'b', 'LineWidth', 1.5); hold on;
plot(time_CP, erp_CP, 'r', 'LineWidth', 1.5);
xlabel('Time (ms)');
ylabel('Amplitude (µV)');
title(sprintf('ERP Comparison (RMSE = %.3f µV, MAE = %.3f µV)', rmse_erp, mae_erp));
legend('EEGLAB','CortiPy');
grid on;
