clear
clc
% clear ReadSharedMemory; 
% 1. Start the producer
NET.addAssembly('System');
proc = System.Diagnostics.Process();
proc.StartInfo.FileName = 'EEG_SharedMemoryProducer.exe';
proc.StartInfo.WorkingDirectory = 'H:\01_Projekte\12_EEG_Analysis_Tool\EEG_Analysis_Tool\Devices\actichamp';
proc.StartInfo.UseShellExecute = false;
proc.Start();

% 2. Wait until shared memory is ready
pause(3);  % give producer time to map shared memory


% Get impedance
imp = GetImpedances();

% 3. Request sampling rate
WriteFs(10000);
% Give producer a moment to apply the sampling rate
disp('Acquisition not ready yet.')
tic
waitFig = uifigure('Name','EEG Setup','Position',[500 500 300 100],'Resize','off');
toc
disp('Acquisition is ready.')
ClearSharedBuffer(); 

% 5. Start reading data
allData = [];
tic
for i = 1:3
    tempData = ReadHybridMemory(20000);  % [channels x newSamples]
    toc
    if ~isempty(tempData)
        allData = [allData; tempData]; 
        fprintf('Now collected %d samples\n', size(allData,1));
    end
    pause(1);
end

% 6. Stop producer cleanly
StopProducer();
% wait until the process exits
while ~proc.HasExited
    pause(0.1);
end


plot(allData(:,33));

