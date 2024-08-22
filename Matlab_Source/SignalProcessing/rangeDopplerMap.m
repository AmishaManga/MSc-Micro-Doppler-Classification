close all;
fclose all;
clear all;
clc

addpath('Libraries');
addpath('Functions');

% Load Radar Params %
radarParams;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read Config file , get data of interest
% If file exists, open file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ini = IniConfig();
[~] = ini.ReadFile('config.ini');
dataPath = string(ini.GetValues('Data','path'));
dataOfInterest = string(ini.GetValues('Data','logFileOfInterest'));

dir_struct = dir( fullfile(dataPath,'*.log') );
if numel(dir_struct) == 0
  disp('No log files')
else
  disp('log file exists')
  file = fullfile(dataPath, dataOfInterest);
  [fileId, message] = fopen(file, 'rb');
    if fileId <0
        error('"failed to open file "%s" because "%s"', file, message);
    end
end

figure;
RDMapCh1 = imagesc(nan);

while feof(fileId) ~= 1
    rawData = uint32(fread(fileId, burstUint32Length, 'uint32'));
    
    % If not a full burst, exit.
    if(length(rawData) < burstUint32Length) 
        disp('Not a full burst - data corruption') 
        break; 
    end
    
    % Reshape raw data into a 1024-by-256 matrix 
    data = reshape(rawData, nextPowOf2Range, noOfDopplerBins)';
    
    RPDSheader = data(:,   1:16);
    Ch1MatrixUint32  = data(:,  17:520);
    Ch2MatrixUint32  = data(:, 521:1024);
    
    % Take RawData Matrix(256x504 unit32) and flatten into a vector format
    % that is 129024(256 times 504) x1
    
    Ch1VectorUint32 = reshape(Ch1MatrixUint32', [], 1);
%     Ch2VectorUint32 = reshape(Ch2MatrixUint32', [], 1);
    
    % Now that the Raw Data is in vector format we can cast back to int16
    % as this is the original specified format so we end up with a vector
    % format that is 258048(256 times 1008) x1
    
    Ch1VectorInt16 = typecast(Ch1VectorUint32, 'int16');
    %Ch2VectorInt16 = typecast(Ch2VectorUint32, 'int16');
    
    % now we want to get back to the shape of 256 by 1008 as this is our
    % Range Doppler Format. We take the transpose as we want range on the x
    % axis
    Ch1MatrixInt16 = reshape(Ch1VectorInt16, noOfRangeBins, noOfDopplerBins)';
    %Ch2MatrixInt16 = reshape(Ch2VectorInt16, noOfRangeBins, noOfDopplerBins)';
    
    % Subtract DC offset from each signal
    % when you take the average of the signal(both AC and DC component)
    % then we get the DC offset, so you just subtract the offset from the signal
    Ch1MatrixInt16 = Ch1MatrixInt16 - mean(mean(Ch1MatrixInt16));
    %Ch2MatrixInt16 = Ch2MatrixInt16 - mean(mean(Ch2MatrixInt16));
    
    % Convert to doubles so that we can use the window function:
    Ch1MatrixDouble = double(Ch1MatrixInt16);
    %Ch2MatrixDouble = double(Ch2MatrixInt16);
    
    % Window Function: Hann Window - Zero Pad up to 1024
    
    Window = repmat(Hann(noOfDopplerBins)', 1, nextPowOf2Range) .* repmat(Hann(nextPowOf2Range), noOfDopplerBins, 1);
    Window = Window / sum(Window);
    
    %fftCh1 = 20*log10(abs(fft2(Ch1MatrixInt16, 256, 1024))); % no window
    fftCh1 = 20*log10(abs(fft2(Ch1MatrixDouble.*Window)));
    fftCh1 = flip(fftCh1,1);
    %fftCh1 = fftshift(fftCh1, 1);
    %fftCh1 = flip(fftCh1,1);
    
    RDMapCh1 = imagesc(fftCh1(:, 1:nextPowOf2Range/2));
    xlabel('Range(bins)'); ylabel('Doppler(bins)');
    yticklabels = 0:20:256;
    yticks = linspace(1, 256, numel(yticklabels));
    set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))

    drawnow();
    pause(0.05);
end