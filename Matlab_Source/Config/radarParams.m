noOfDopplerBins = 256;
noOfRangeBins = 1008;
nextPowOf2Range = 1024; % to zero pad the range fft in the range fft
burstUint32Length = 256*1024;
pri = 0.0014;
fastAlpha = 0.9;
slowAlpha = 0.01;
threshold = 13; % detection map
pfa = 10^-5; % desired probability false alarm rate
pulseRepetitionInterval = 625; % in Hertz

%% Spectrogram Config:
WinL = 128;
overLap = 8;   
nfft = 256; 
dbdown =80; 

%% Humans:
% nReferenceCellsRange = 4; % no. of reference cells for Range CFAR 
% nGuardCellsRange = 6; %4;
% nReferenceCellsDoppler = 10; % 8; % no. of reference cells for Doppler CFAR 
% nGuardCellsDoppler = 16; % 14;

% %% Drones:
nReferenceCellsRange = 4; % no. of reference cells for Range CFAR 
nGuardCellsRange = 6; %4;
nReferenceCellsDoppler =8; % 8; % no. of reference cells for Doppler CFAR 
nGuardCellsDoppler = 10; % 14;