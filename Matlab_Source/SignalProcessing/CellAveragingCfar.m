%%
% This script plots the range doppler map, range-doppler cfar as well as
% the individual range and doppler cfar maps.
% Uses data from a log file.
%%
close all;
fclose all;
clear all;
clc

addpath('../Libraries');
addpath('../Functions');
addpath('../Config');

% Load Radar Params %
radarParams;

maxRange = 121;
stepSize = 20;
maxRangeBin = round(maxRange/ 1.875);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read Config file , get data of interest
% If file exists, open file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get data of interest
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
end

validData = true;
% figure;
%Threshold = plot(nan(1, 1008)); hold on;
%Ch1Signal = plot(nan(1, 1008));
detectionMatrixRangeDopplerCombined = zeros(256,1008);

while feof(fileId) ~= 1
    rawData = uint32(fread(fileId, burstUint32Length, 'uint32'));
    
    % If not a full burst, exit.
    if(length(rawData) < burstUint32Length) 
        disp('Not a full burst - data corruption');
        validData = false;
        %break; 
    end
    
    % Reshape raw data into a 1024-by-256 matrix 
    data = reshape(rawData, nextPowOf2Range, noOfDopplerBins)';

    RPDSheader = data(:,   1:16);
    Ch1MatrixUint32  = data(:,  17:520);
    Ch2MatrixUint32  = data(:, 521:1024);

    % Take RawData Matrix(256x504 unit32) and flatten into a vector format
    % that is 129024(256 times 504) x1

    Ch1VectorUint32 = reshape(Ch1MatrixUint32', [], 1);
    Ch2VectorUint32 = reshape(Ch2MatrixUint32', [], 1);

    % Now that the Raw Data is in vector format we can cast back to int16
    % as this is the original specified format so we end up with a vector
    % format that is 258048(256 times 1008) x1

    Ch1VectorInt16 = typecast(Ch1VectorUint32, 'int16');
    Ch2VectorInt16 = typecast(Ch2VectorUint32, 'int16');

    % now we want to get back to the shape of 256 by 1008 as this is our
    % Range Doppler Format. We take the transpose as we want range on the x
    % axis
    Ch1MatrixInt16 = reshape(Ch1VectorInt16, noOfRangeBins, noOfDopplerBins)';
    Ch2MatrixInt16 = reshape(Ch2VectorInt16, noOfRangeBins, noOfDopplerBins)';

    % Convert to doubles so that we can use the window function:
    Ch1MatrixDouble = double(Ch1MatrixInt16);
    Ch2MatrixDouble = double(Ch2MatrixInt16);

    % Subtract DC offset from each signal
    % when you take the average of the signal(both AC and DC component)
    % then we get the DC offset, so you just subtract the offset from the signal
    Ch1MatrixDouble = Ch1MatrixDouble - repmat(mean(Ch1MatrixDouble, 1), size(Ch1MatrixDouble, 1), 1);
    Ch1MatrixDouble = Ch1MatrixDouble - mean(Ch1MatrixDouble);
    
    Ch2MatrixDouble = Ch2MatrixDouble - repmat(mean(Ch2MatrixDouble, 1), size(Ch2MatrixDouble, 1), 1);
    Ch2MatrixDouble = Ch2MatrixDouble - mean(Ch2MatrixDouble);

    % Window Function: Hann Window - Zero Pad up to 1024

    Window = repmat(Hann(noOfDopplerBins)', 1, nextPowOf2Range) .* repmat(Hann(nextPowOf2Range), noOfDopplerBins, 1);
    Window = Window / sum(Window);

    %fftCh1 = 20*log10(abs(fft2(Ch1MatrixInt16, 256, 1008))); % no window
    fftCh1 = abs(fft2(Ch1MatrixDouble.*Window));
    fftCh1 = flip(fftCh1,1); % flipping here because we want to see 0 doppler at 0 range
    fftCh1Shift = fftshift(fftCh1, 1);

    % Pass to CFAR Detection function in Range
    thresholdMatrixRange = 10*log10(CfarDetectionLoopRange(fftCh1Shift,pfa,nReferenceCellsRange,nGuardCellsRange));
    detectionMatrixRangeBinary = 20*log10(fftCh1Shift) > thresholdMatrixRange;
    detectionMatrixRange =  detectionMatrixRangeBinary .* (20*log10(fftCh1Shift));

    % Pass to CFAR Detection function in Doppler
    thresholdMatrixDoppler = 10*log10(CfarDetectionLoopDoppler(fftCh1Shift,pfa,nReferenceCellsDoppler,nGuardCellsDoppler));
    detectionMatrixDopplerBinary = 20*log10(fftCh1Shift) > thresholdMatrixDoppler;
    detectionMatrixDoppler = detectionMatrixDopplerBinary .* (20*log10(fftCh1Shift));

    % Pass to CFAR Detection function in Range and Doppler
    detectionMatrixRangeDoppler = detectionMatrixRangeBinary .* detectionMatrixDopplerBinary;
    
    % Plot this for a build up of detections in order to see a track:
    %detectionMatrixRangeDopplerCombined = detectionMatrixRangeDopplerCombined + detectionMatrixRangeDoppler;
    %detectionMatrixRangeDopplerCombined = detectionMatrixRangeDopplerCombined > 0; 

    % Plot Range Doppler Map FFT Shifted
    ax1 = subplot(2,2,1);
    RangeDopplerMapfftshift = imagesc(20*log10(fftCh1Shift(:, 1:(maxRangeBin))));
    title('Range Doppler Map')
    xlabel('Range (m)'); 
    ylabel('Doppler (m/s)');
    xticklabels = 0:stepSize:maxRange;
    yticklabels = -4.6:2.3:4.6;    
    set(ax1,'YTick', linspace(1, 256, numel(yticklabels)), 'YTickLabel', flipud(yticklabels(:)), ...
    'XTick',linspace(1, maxRangeBin, numel(xticklabels)), 'XTickLabel', xticklabels ); 
%     colorbar;

    % Plot CA CFAR Detection Map : Range and Doppler
    ax2 = subplot(2,2,2);
    RangeDopplerCfar = imagesc(detectionMatrixRangeDoppler(:, 1:maxRangeBin));
    title('Range & Doppler CA CFAR Map')
    xlabel('Range (m)'); 
    ylabel('Doppler (m/s)');
    xticklabels = 0:stepSize:maxRange;
    yticklabels = -4.6:2.3:4.6;
    set(ax2,'YTick', linspace(1, 256, numel(yticklabels)), 'YTickLabel', flipud(yticklabels(:)), ...
    'XTick',linspace(1, maxRangeBin, numel(xticklabels)), 'XTickLabel', xticklabels ); 
%     colorbar;

    % Plot Range CFAR
    ax3 = subplot(2,2,3);
    RangeCfar = imagesc(detectionMatrixRange(:, 1:maxRangeBin));
    title('Range CA-CFAR Map')
    xlabel('Range (m)'); 
    ylabel('Doppler (m/s)');
    xticklabels = 0:stepSize:maxRange;
    yticklabels = -4.6:2.3:4.6;
    set(ax3,'YTick', linspace(1, 256, numel(yticklabels)), 'YTickLabel', flipud(yticklabels(:)), ...
    'XTick',linspace(1, maxRangeBin, numel(xticklabels)), 'XTickLabel', xticklabels ); 
%     colorbar;
    
    % Plot Doppler CFAR
    ax4 = subplot(2,2,4);
    DopplerCfar = imagesc(detectionMatrixDoppler(:, 1:maxRangeBin));
    title('Doppler CA-CFAR Map')
    xlabel('Range (m)'); 
    ylabel('Doppler (m/s)');
    xticklabels = 0:stepSize:maxRange;
    yticklabels = -4.6:2.3:4.6;
    set(ax4,'YTick', linspace(1, 256, numel(yticklabels)), 'YTickLabel', flipud(yticklabels(:)), ...
    'XTick',linspace(1, maxRangeBin, numel(xticklabels)), 'XTickLabel', xticklabels ); 
%     colorbar;
    
    drawnow();
    pause();
    
end
