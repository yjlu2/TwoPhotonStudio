function studio_export_heatmap_fig(matPath, figPath)
% Create an editable MATLAB figure for the trial-aligned response heatmap.
S=load(matPath); t=double(S.time(:)); heat=double(S.sortedMeanZ);
fig=figure('Visible','off','Color','w','Position',[100 100 1050 780]);
layout=tiledlayout(fig,3,1,'TileSpacing','compact','Padding','compact');
stim=nexttile(layout,1); hold(stim,'on');
stop=min(double(S.stimulusDuration),double(S.postSeconds));
patch(stim,[0 stop stop 0],[0 0 1 1],[1 .18 .22],'FaceAlpha',.40,'EdgeColor',[.9 .1 .12]);
plot(stim,[t(1) 0 0 stop stop t(end)],[0 0 1 1 0 0],'Color',[.9 .1 .12],'LineWidth',1.2);
ylim(stim,[-.05 1.05]); xlim(stim,[t(1) t(end)]); set(stim,'YTick',[],'XTickLabel',[]); ylabel(stim,'Stimulus'); box(stim,'on');
trace=nexttile(layout,2); population=mean(double(S.meanZ),1,'omitnan');
plot(trace,t,population,'k','LineWidth',1.2); xline(trace,0,'Color',[.9 .1 .12]);
xlim(trace,[t(1) t(end)]); ylabel(trace,'Mean z-score'); set(trace,'XTickLabel',[]); box(trace,'on');
ax=nexttile(layout,3); imagesc(ax,t,1:size(heat,1),heat); set(ax,'YDir','reverse');
xline(ax,0,'k-'); xlabel(ax,'Time from stimulus onset (s)'); ylabel(ax,'Cell ROIs (response-sorted)');
finite=abs(heat(isfinite(heat))); if isempty(finite), lim=1; else, lim=max(.5,prctile(finite,98)); end
clim(ax,[-lim lim]); colormap(ax,'turbo'); cb=colorbar(ax); ylabel(cb,'dF/F z-score');
title(stim,sprintf('%d cells | %d trials | baseline -%.3g to 0 s',size(heat,1),size(S.trialZ,2),double(S.baselineSeconds)));
% Keep the generated figure hidden during batch export, but save it with a
% visible state so openfig/uiopen displays it in an interactive MATLAB session.
set(fig,'Visible','on');
savefig(fig,figPath); close(fig);
end
