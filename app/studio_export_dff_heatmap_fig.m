function studio_export_dff_heatmap_fig(matPath, figPath)
% Create an editable no-sync dF/F trace heatmap with MATLAB parula.
S=load(matPath);
t=double(S.time(:)); heat=double(S.heatmap);
fig=figure('Visible','off','Color','w','Position',[100 100 1250 720]);
ax=axes(fig);
imagesc(ax,t,1:size(heat,1),heat);
set(ax,'YDir','normal'); axis(ax,'tight');
colormap(ax,parula(256)); cb=colorbar(ax);
cb.Label.String='Robust normalized dF/F';
clim(ax,[-1 1]);
xlabel(ax,'Time (s)'); ylabel(ax,'ROI');
title(ax,sprintf('dF/F trace heatmap | %d ROIs',size(heat,1)));
box(ax,'on');
% savefig preserves the figure Visible property. Store it as visible so the
% exported FIG opens normally with uiopen, openfig, or a double-click.
set(fig,'Visible','on');
savefig(fig,figPath); close(fig);
end
