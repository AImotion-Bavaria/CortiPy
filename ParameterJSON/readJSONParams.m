function Params = readJSONParams(file)
[~, name, ~] = fileparts(file);
% Step 1: Read JSON text from file
jsonText = fileread(file);

% Step 2: Decode JSON text to struct
rawData = jsondecode(jsonText);

% Step 3: Convert array of structs into struct of arrays

%Params.(name) = rawData.(name);
Params = rawData.(name);