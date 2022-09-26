% generate image datastore
imds = imageDatastore('spgramImages', ...
    'IncludeSubfolders',false,...
    "LabelSource","foldernames");

L=length(imds.Files);
pat="\" + ("A"|"W"|"F"|"N"|"T"|"E"|"L")+digitsPattern+".png";

for i=1:L
%     fileName=extract(imds.Files{i},pat);
    imds.Labels(i)=imds.Files{i}(50);
    if (mod(i,100)==0)
        disp(i);
    end
end

save("spgramImds.mat","imds");