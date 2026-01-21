function electrodeMap = loadElectrodes()
    % Try to find electrodes.json on MATLAB path or cwd
    jsonName = 'electrodes.json';
    fullpath = which(jsonName);
    if isempty(fullpath)
        % fall back to cwd
        fullpath = fullfile(pwd, jsonName);
        if ~isfile(fullpath)
            error('Could not locate %s on MATLAB path or in the current folder (%s).', ...
                  jsonName, pwd);
        end
    end

    txt = fileread(fullpath);
    electrodeMap = jsondecode(txt);
end