function apply_studio_manual_Fneu(destination,patchFile)
% Apply native TwoPhotonStudio fluorescence extraction to the selected rows.
D=load(destination); P=load(patchFile); old=D;
rows=double(P.roiRows(:)); S=D.manualNeuropil;
assert(isequal(rows,double(S.roiRows(:))),'Manual ROI pairing changed.');
assert(numel(rows)==size(P.Fneu,1) && size(P.Fneu,2)==size(D.F,2));
for i=1:numel(rows)
    [y,x]=find(S.neuropilMasks(:,:,i));
    expected=sort((y-1)*size(S.neuropilMasks,2)+(x-1));
    assert(isequal(expected,double(P.neuropilIndices{i}(:))),'Mask pixels mismatch.');
end
names={'Fneu','Fc','F0','dff','spks'};
other=setdiff(1:size(D.F,1),rows);
for i=1:numel(names)
    name=names{i}; D.(name)(rows,:)=cast(P.(name),'like',D.(name));
    assert(isequaln(D.(name)(other,:),old.(name)(other,:)));
end
for i=1:numel(rows)
    r=rows(i);
    if iscell(D.stat)
        D.stat{r}.neuropil_mask=int64(P.neuropilIndices{i}(:)');
        D.stat{r}.std=double(P.roiStd(i)); D.stat{r}.skew=double(P.roiSkew(i));
    else
        D.stat(r).neuropil_mask=int64(P.neuropilIndices{i}(:)');
        D.stat(r).std=double(P.roiStd(i)); D.stat(r).skew=double(P.roiSkew(i));
    end
end
D.manualNeuropil.originalFneu=old.Fneu(rows,:);
D.manualNeuropil.traceStatus='Fneu re-extracted from registered movie using exact manually drawn union masks; Fc, F0, dff and spks updated by TwoPhotonStudio.';
D.manualNeuropil.nativeResult=P.nativeResult;
D.manualNeuropil.reextractedAt=char(datetime('now'));
D.ops.manual_neuropil_rows=rows;
D.ops.manual_neuropil_extraction='TwoPhotonStudio / Suite2p 0.10.3 CPU';
assert(isequaln(D.F,old.F) && isequaln(D.iscell,old.iscell));
backup=fullfile(fileparts(destination),['Fall-2_before_manual_Fneu_' char(datetime('now','Format','yyyyMMdd_HHmmss')) '.mat']);
copyfile(destination,backup);
temp=[destination '.partial.mat']; save(temp,'-struct','D','-v7.3');
check=load(temp,'Fneu','manualNeuropil');
assert(isequaln(check.Fneu,D.Fneu));
assert(isequal(check.manualNeuropil.polygons,S.polygons));
assert(isequal(check.manualNeuropil.neuropilMasks,S.neuropilMasks));
movefile(temp,destination,'f');
fprintf('Updated %d ROI rows: %s\nBackup: %s\n',numel(rows),mat2str(rows'),backup);
end
