function ConnectToActiCHamp()

    % --- Start EEG Shared Memory Producer ---

        NET.addAssembly('System');
        proc = System.Diagnostics.Process();
        proc.StartInfo.FileName = fullfile(pwd,'Devices','actichamp','EEG_SharedMemoryProducer.exe');
        proc.StartInfo.WorkingDirectory = fullfile(pwd,'Devices','actichamp');
        proc.StartInfo.UseShellExecute = false;
        proc.Start();
        disp('ActiCHamp EEG producer started.');

    
end
