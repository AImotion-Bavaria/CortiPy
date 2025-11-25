function toggleEEGProducerActiChamp()
% TOGGLEEEGPRODUCER Starts or stops the ActiCHamp EEG Shared Memory Producer.
%   - If no process exists, starts it
%   - If process exists and running, stops it

    persistent proc

    % Case 1: process not created or exited → start it
    if isempty(proc) || ~isvalid(proc) || proc.HasExited
        NET.addAssembly('System');
        proc = System.Diagnostics.Process();
        proc.StartInfo.FileName = fullfile(pwd,'Devices','actichamp','EEG_SharedMemoryProducer.exe');
        proc.StartInfo.WorkingDirectory = fullfile(pwd,'Devices','actichamp');
        proc.StartInfo.UseShellExecute = false;
        proc.Start();
        disp('ActiCHamp EEG producer started.');
        return;
    end

    % Case 2: process exists & running → stop it
    if ~proc.HasExited
        try
            % Call your SDK stop function if available
            StopProducer();  % optional, if you have it
        catch
            warning('StopProducer() not available, trying to kill process.');
        end
        % Wait for exit or kill if necessary
        while ~proc.HasExited
            pause(0.1);
        end
        disp('ActiCHamp EEG producer stopped.');
        proc = [];
        return;
    end
end
