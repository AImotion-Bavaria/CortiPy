function UpdateChannelImpedances(panel)
    % Get impedance values (vector)
    impValuesTemp = GetImpedances();  % in Ohm
    impValues = impValuesTemp / 1000; % in KOhm
    impValues(2) = [];
    % Loop over all impedance edit fields in the panel
    for k = 1:numel(impValues)
        ef = findobj(panel, 'Tag', sprintf('Impedance_%d', k));
        if ~isempty(ef)
            ef.Value = impValues(k);
        end
    end
end
