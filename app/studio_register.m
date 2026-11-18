function studio_register(configPath)
% Stream the SAME DFT algorithm as reg_stacks.m.
% Records every frame's shift; excludes Fourier wraparound in Python.
c=jsondecode(fileread(configPath)); addpath(c.code);
assert(exist('dftregistration','file')==2,'dftregistration.m is missing');
files=cellstr(string(c.inputs)); counts=zeros(numel(files),1);
for k=1:numel(files), counts(k)=numel(imfinfo(files{k})); end
n=sum(counts);
% A fixed first-frame template matches reg_stacks(Images, refImg).
refImg=double(imread(files{1},1)); refFFT=fft2(refImg);
yoff=zeros(n,1); xoff=yoff; regError=yoff;
f=fopen(c.output,'w','ieee-le'); assert(f>=0); cleanup=onCleanup(@() fclose(f));
t=0;
for k=1:numel(files)
for page=1:counts(k)
    t=t+1; im=double(imread(files{k},page));
    [o,G]=dftregistration(refFFT,fft2(im),10);
    aligned=abs(ifft2(G));
    % Suite2p int16 binary, row-major; same intensity units as uint16/2 import.
    fwrite(f,int16(min(32767,max(0,floor(aligned/2))))','int16');
    regError(t)=o(1); yoff(t)=o(3); xoff(t)=o(4);
    if mod(t,100)==0 || t==n, fprintf('DFT registered %d/%d\n',t,n); end
end
end
clear cleanup;
save(c.metadata,'refImg','yoff','xoff','regError','-v7');
end
