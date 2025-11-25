function eegProducerActiChamp(action, channelPanel)
% EEGPRODUCERACTICHAMP Manage ActiCHamp producer + impedance timer
%   eegProducerActiChamp('start', channelPanel)
%   eegProducerActiChamp('stop')
%   eegProducerActiChamp('stop-timer')

persistent proc impedanceTimer panelRef

if nargin < 2
    channelPanel = [];
end


switch lower(action)
    case 'start'
        panelRef = channelPanel;

        % --- Start EEG producer if needed ---
        if isempty(proc) || ~isvalid(proc) || proc.HasExited
            NET.addAssembly('System');
            proc = System.Diagnostics.Process();
            proc.StartInfo.FileName = fullfile(pwd,'Devices','actichamp','EEG_SharedMemoryProducer.exe');
            proc.StartInfo.WorkingDirectory = fullfile(pwd,'Devices','actichamp');
            proc.StartInfo.UseShellExecute = false;
            proc.Start();
            disp('ActiCHamp EEG producer started.');
            % Wait until shared memory exists (max 1 sec)
            % pause(5); 
            % SetActiveElectrodes(false);
            % pause(0.5);
            %         SetActiveElectrodes(true);
            %         break;
            %     catch
            %         pause(0.1);
            %     end
            % end
        end

        % --- Start impedance timer if needed ---
        if isempty(impedanceTimer) || ~isvalid(impedanceTimer)
            impedanceTimer = timer('ExecutionMode','fixedSpacing', ...
                'Period',1, ...
                'TimerFcn',@(~,~)safeUpdate());
            start(impedanceTimer);
            disp('Impedance timer started.');
        end

    case 'stop-timer'
        % --- Stop impedance timer only ---
        if ~isempty(impedanceTimer) && isvalid(impedanceTimer)
            stop(impedanceTimer);
            delete(impedanceTimer);
            impedanceTimer = [];
            disp('Impedance timer stopped.');
        end

    case 'stop'
        % --- Stop timer ---
        if ~isempty(impedanceTimer) && isvalid(impedanceTimer)
            stop(impedanceTimer);
            delete(impedanceTimer);
            impedanceTimer = [];
            disp('Impedance timer stopped.');
        end

        % --- Stop producer ---
        if ~isempty(proc) && isvalid(proc) && ~proc.HasExited
            try StopProducer(); catch, proc.Kill(); end
            proc.WaitForExit();
            disp('ActiCHamp EEG producer stopped.');
        end
        proc = [];
end

% --- Local helper for safe UI updates ---
    function safeUpdate()
        try
            if isvalid(panelRef)
                UpdateChannelImpedances(panelRef);
            end
        catch
            % avoid flooding warnings if UI closed
        end
    end
end
