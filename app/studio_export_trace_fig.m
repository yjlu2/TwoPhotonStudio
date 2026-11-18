function studio_export_trace_fig(matPath, figPath)
% Build an editable MATLAB figure from a selected-ROI export bundle.
S = load(matPath);
names = cellstr(string(S.displayTraceNames));
t = double(S.time(:));
if isfield(S,'roiIndices')
    indices = double(S.roiIndices(:)); colors = double(S.roiColors);
else
    indices = double(S.roiIndex); colors = [1 .86 .25];
end
styles = struct('F','-','Fneu','--','Fc','-.','dff',':','spks','-');
fig = figure('Visible','off','Color','none','Position',[100 100 1350 500]);
ax = axes(fig,'Color','none'); hold(ax,'on');
for r = 1:numel(indices)
    for k = 1:numel(names)
        name = char(names{k});
        if ~isfield(S,name), continue; end
        values = double(S.(name));
        if isvector(values), y=values(:); else, y=values(r,:).'; end
        if logical(S.displayNormalized)
            good=isfinite(y);
            if any(good), lo=min(y(good)); y=(y-lo)/max(max(y(good))-lo,1e-9); end
        end
        marker='none';
        if strcmp(name,'spks'), marker='.'; end
        plot(ax,t,y,'Color',colors(r,:),'LineStyle',styles.(name),'Marker',marker,...
            'MarkerIndices',1:max(1,round(numel(t)/120)):numel(t),'LineWidth',0.85,...
            'DisplayName',sprintf('ROI %d %s',indices(r),name));
    end
end
if isfield(S,'stimTimes') && isfield(S,'displayShowStim') && logical(S.displayShowStim)
    starts=double(S.stimTimes(:)); ends=[];
    if isfield(S,'stimEndTimes'), ends=double(S.stimEndTimes(:)); end
    yl=ylim(ax);
    for index=1:numel(starts)
        onset=starts(index);
        if index<=numel(ends)
            patch(ax,[onset ends(index) ends(index) onset],[yl(1) yl(1) yl(2) yl(2)],...
                [1 0.18 0.22],'FaceAlpha',0.30,'EdgeColor',[1 0.25 0.28],...
                'LineWidth',0.75,'HandleVisibility','off');
        end
        xline(ax,onset,'--','Color',[1 0.25 0.28],'LineWidth',0.75,'HandleVisibility','off');
    end
end
xlabel(ax,'Time (s)');
if logical(S.displayNormalized), ylabel(ax,'Normalized display'); else, ylabel(ax,'Native units / dF/F'); end
title(ax,sprintf('%d selected ROI trace(s) | color = ROI, line style = metric',numel(indices)));
legend(ax,'Location','eastoutside','Box','off'); box(ax,'on'); set(ax,'XColor','k','YColor','k');
savefig(fig,figPath); close(fig);
end
