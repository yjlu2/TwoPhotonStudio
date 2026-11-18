function [NewImg output] = reg_stacks(Images,refImg)
usfac = 10;
NewImg = zeros(size(Images));
tif_Num = size(Images,3);
hwait = waitbar(0,'Now Registing FOV. Please Wait...');
for ii = 1:size(Images,3);
    img = Images(:,:,ii);
    [output ,Greg] = dftregistration(fft2(refImg),fft2(img),usfac);
    NewImg(:,:,ii) = abs(ifft2(Greg));
    waitbar(ii/tif_Num)
end
close(hwait)
end