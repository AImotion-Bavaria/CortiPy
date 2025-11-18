function toggleImpedanceTimer(channelPanel)
% TOGGLEIMPEDANCETIMER Toggle impedance timer on/off.
%   - If no timer exists, creates and starts it.
%   - If timer exists and is running, stops & deletes it.
%   - If timer exists but is stopped, restarts it.

persistent impedanceTimer

% Case 1: no timer yet or invalid → create & start
if isempty(impedanceTimer) || ~isvalid(impedanceTimer)
    impedanceTimer = timer( ...
        'ExecutionMode','fixedSpacing', ...
        'Period',1, ...
        'TimerFcn', {@timerCallback, channelPanel}, ...
        'Name','ImpedanceTimer');
    start(impedanceTimer);
    disp('Impedance timer started.');
    return;
end

% Case 2: timer exists & running → stop & delete
if strcmp(impedanceTimer.Running,'on')
    stop(impedanceTimer);
    delete(impedanceTimer);
    impedanceTimer = [];
    disp('Impedance timer stopped.');
    return;
end

% Case 3: timer exists but not running → restart
if strcmp(impedanceTimer.Running,'off')
    start(impedanceTimer);
    disp('Impedance timer restarted.');
end


end

function timerCallback(~,~,channelPanel)
try
UpdateChannelImpedances(channelPanel);
catch ME
warning('Timer callback error');
end
end
