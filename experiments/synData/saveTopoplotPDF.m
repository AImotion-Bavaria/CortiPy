function saveTopoplotPDF(ampData, chanlocs, titleStr, filename, unitStr)
    % Ensure output folder exists
    [folder, ~, ~] = fileparts(filename);
    if ~exist(folder, 'dir')
        mkdir(folder);
    end

    % Create figure
    fig = figure('Visible','off','Color','w', ...
             'Units','centimeters', ...
             'Position',[0 0 20 20]);   % FIXED 5×5 cm square figure

    % Create axes
    ax = axes('Parent', fig, 'Color', 'w'); 
    axes(ax);

    % Plot topography with white background outside head
    topoplot(ampData, chanlocs, ...
        'electrodes','on', ...
        'maplimits', [min(ampData) max(ampData)], ...     % set your real scale
        'style','fill', ...       % head filled with colormap
        'headrad',0.5, ...
        'whitebk','on');          % sets outside head to white

    % --- Bigger text everywhere ---
    ax.FontSize = 48;       % electrode label sizing + axes text
    title(titleStr, 'FontSize', 48, 'FontWeight', 'bold');

    % Add colorbar with unit
    c = colorbar;
    c.Label.String = unitStr;   
    c.Label.FontSize = 48;

    % Export as vectorized PDF
    exportgraphics(fig, filename, 'ContentType', 'vector');

    close(fig);
end
