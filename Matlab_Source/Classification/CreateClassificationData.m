close all;
close all hidden;
fclose all;
clear all;
clc;
format compact;
format short;

%% Add Paths
addpath('../Libraries');
addpath('../Functions');
addpath('../Config');
radarParams;

%% Flags
validData = true;
rangeBin = 1;
j = 1;
burstCount = 0;
longerSignal = []; 

%% Get dataSet of interest
[dataOfInterest,dataPath] = uigetfile({'*.log';'*.m'}, ...
                            'Select a file','E:\MastersData\');
if isequal(dataOfInterest,0)
   disp('User selected Cancel');
else
   disp(['User selected ', fullfile(dataPath,dataOfInterest)]);
   file = fullfile(dataPath, dataOfInterest);
   [fileId, message] = fopen(file, 'rb');
    if fileId <0
        error('"failed to open file "%s" because "%s"', file, message);
    end
    addpath(dataPath);
    Bin = rangeBinData(dataOfInterest);
end

longerSignal = [];
longerSignalM = [];
longerSignalP = [];

while true
  
%%Process
timestamp_bytes = uint8(fread(fileId, 12, 'uint8'));    
rawData = uint32(fread(fileId, burstUint32Length, 'uint32'));
% timestamp_str = char(timestamp_bytes');

if(length(rawData) < burstUint32Length) 
    disp('Not a full burst - data corruption');
    validData = false;
    %break; 
end
    
dataRaw = reshape(rawData, nextPowOf2Range, noOfDopplerBins)';

RPDSheader = dataRaw(:,   1:16);
Ch1MatrixUint32  = dataRaw(:,  17:520);
Ch2MatrixUint32  = dataRaw(:, 521:1024);

%Added
% rangeFFT1 = doRangeFFT(Ch1MatrixUint32,noOfRangeBins,noOfDopplerBins);
% rangeFFT2 = doRangeFFT(Ch2MatrixUint32,noOfRangeBins,noOfDopplerBins); 
% rangeFFT = rangeFFT1 + rangeFFT2;

%Channel 1
% rangeFFT = doRangeFFT(Ch1MatrixUint32,noOfRangeBins,noOfDopplerBins);

%Channel 2
rangeFFT = doRangeFFT(Ch2MatrixUint32,noOfRangeBins,noOfDopplerBins); 

binPlusOne = Bin(j) + 1;
binMinusOne = Bin(j) - 1;
rangeBinMinusOneSignal = rangeFFT(:,binMinusOne);
rangeBinPlusOneSignal = rangeFFT(:,binPlusOne);

selectedRangeBinSignal = rangeFFT(:,Bin(j));

% stitch all bursts together into one long signal
longerSignal = cat(1,longerSignal,selectedRangeBinSignal);
longerSignalM = cat(1,longerSignalM,rangeBinMinusOneSignal);
longerSignalP = cat(1,longerSignalP,rangeBinPlusOneSignal);

longerSignalM = longerSignalM - mean(longerSignalM);
longerSignal = longerSignal - mean(longerSignal);
longerSignalP = longerSignalP - mean(longerSignalP);

burstCount = burstCount +1;

 %% Draw the Spectrogram
if burstCount == 20
         
    STFTM = Specgram(longerSignal,    128 , 8, 256);
    STFTMM = Specgram(longerSignalM,  128 , 8, 256);
    STFTMP = Specgram(longerSignalP,  128 , 8, 256);

%     addedSTFM = abs(STFTMM) + abs(STFTM) + abs(STFTMP);
    normSTFT = abs(STFTM);
%       normSTFT= addedSTFM;

    normalisedSTFM = max(max(normSTFT));
    SpecData = imagesc(20*log10(normSTFT./normalisedSTFM));
    set(gca,'CLim', [-30 -10]);
    colorbar;
    colormap(jet);

    title("Spectrogram");
    ylabel('Velocity(m/s)'); 
    xlabel('Time (s)'); 
    colormap(jet);
    yticklabels = -4.6:2.3:4.6; 
    xticklabels = 0:1:pri*burstCount*noOfDopplerBins; 
    set(gca, 'YTick', linspace(1, nfft, numel(yticklabels)), 'YTickLabel', flipud(yticklabels(:)), ...
    'XTick',linspace(1, width(STFTM), numel(xticklabels)), 'XTickLabel', round(xticklabels) );
    
    % Create Training and Test Set
    button = classifyData(20*log10(normSTFT./normalisedSTFM));
    if strcmpi(button, 'Exit')
         break;
    end

    longerSignal = [];
    longerSignalM = [];
    longerSignalP = [];
    STFTM = [];
    STFTMM = [];
    STFTMP = [];  
    burstCount =0;
%     pause;

end  % if burst count   

j = j+1;
drawnow;


end % While Loop

close all;
close all hidden;
fclose all;
clear all;
clc;
