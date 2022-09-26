clearvars
clc
close all

%add voicebox path
addpath('D:\Proiecte\ODIN112\detectie_emotii\sap-voicebox-master\voicebox')

%% config parameters
rootPath='D:\Proiecte\ODIN112\detectie_emotii\'; %this is the root path, where all files are stored
fs=16000;
frameLength=16000;
inputData=struct;
stride=160;
ninc=0.004*fs;           % Frame increment for BW=200 Hz (in samples)
nwin=0.016*fs;              % Frame length (in samples)
win=hamming(nwin);        % Analysis window
p=0.5*fs*sum(win.^2);  
map=colormap('jet');

%% parse file from folders
files=dir([rootPath 'corpora\EMODB\*.wav']);
numFiles=length(files);
k=1;
sumMin=0;
sumMax=0;
for i=1:numFiles
    [y,fs]=audioread([files(i).folder '\' files(i).name]);
    emotionLabel=files(i).name(6);
    wavLength=length(y);
    idx=0;
    while ((idx+frameLength)<wavLength)
        inputData(k).emotion=emotionLabel;
        inputData(k).waveform=y(idx+1:idx+frameLength)';
        sf=abs(v_rfft(v_enframe(inputData(k).waveform,win,ninc),nwin,2)).^2/p;
        sf(sf==0)=1e-15;
        [t,f,b]=v_spgrambw(sf,[fs/ninc 0.5*(nwin+1)/fs fs/nwin],'jdfhHc',118,fs/2,[-150,-27]);
        map=colormap('jet');
        minv = min(b(:));
        maxv = max(b(:));
        ncol = size(map,1);
        s = round(1+(ncol-1)*(b-minv)/(maxv-minv));
        specImage=ind2rgb(s,map);
        specImage=rot90(specImage);
        specImage=imresize(specImage,[256,256]);
        imwrite(specImage,[rootPath '\spgramImages\' emotionLabel num2str(k) '.png']);
%         inputData(k).spectrogram=specImage;
        if (min(min(b))==-Inf)
            disp('kk');
            disp(k);
        end
        sumMin=min(min(b))+sumMin;
        sumMax=max(max(b))+sumMax;
        k=k+1;
        idx=idx+stride;
    end
    if (idx<wavLength)
        inputData(k).emotion=emotionLabel;
        inputData(k).waveform=[y(idx:end)' zeros(1,frameLength-length(y(idx:end)))];
        sf=abs(v_rfft(v_enframe(inputData(k).waveform,win,ninc),nwin,2)).^2/p; 
        sf(sf==0)=1e-15;
        [t,f,b]=v_spgrambw(sf,[fs/ninc 0.5*(nwin+1)/fs fs/nwin],'jdfhHc',118,fs/2,[-150,-27]);
        map=colormap('jet');
        minv = min(b(:));
        maxv = max(b(:));
        ncol = size(map,1);
        s = round(1+(ncol-1)*(b-minv)/(maxv-minv));
        specImage=ind2rgb(s,map);
        specImage=rot90(specImage);
        specImage=imresize(specImage,[256,256]);
        imwrite(specImage,[rootPath '\spgramImages\' emotionLabel num2str(k) '.png']);

%         inputData(k).spectrogram=specImage;
        if (min(min(b))==-Inf)
            disp('kk');
            disp(k);
        end
        sumMin=min(min(b))+sumMin;
        sumMax=max(max(b))+sumMax;
        k=k+1;
    end
    if (mod(i,10)==0)
        disp(i);
        disp(k);
    end
end

numExamples=k-1;
averageMin=sumMin/numExamples;
averageMax=sumMax/numExamples;

