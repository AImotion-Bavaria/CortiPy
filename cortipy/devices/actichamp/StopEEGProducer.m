function StopEEGProducer()
    persistent proc

    if isempty(proc) || ~isvalid(proc)
        return;
    end

    % signal the producer to stop via MEX
    try
        StopProducer();  % sets stopRequested in shared memory
    catch
        warning('StopProducer MEX call failed.');
    end

    % wait until the process really exits
    if isvalid(proc) && ~proc.HasExited
        while ~proc.HasExited
            pause(0.1);
        end
    end

    proc.Dispose();
    proc = [];
    disp('EEG producer fully stopped.');
end
