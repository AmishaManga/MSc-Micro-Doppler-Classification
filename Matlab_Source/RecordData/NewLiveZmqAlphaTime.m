%%
% This script displays detections using the alpha or progressive
% filter method. Uses live data from ZMQ stream.
%
%%
close all;
fclose all;
clear all;
clc

%% Also changed the function here...DetectionThreshold
%% Params
maxRange = 150;
stepSize = 10;
maxRangeBin = round(maxRange/ 1.875);

%%
% fast map:
dBScaleUpperFast = 35;
dBScaleLowerFast = -20;

% slow map:
dBScaleUpperSlow = 35;
dBScaleLowerSlow = -20;

% difference map:
dBScaleUpperDiff = 30;
dBScaleLowerDiff = -5;

% detection map:
dBScaleUpperDet = 30;
dBScaleLowerDet = -5;

%%
try
    socket.close();
catch
end

%%
addpath('../Libraries');
addpath('../Functions');
addpath('../Config');

% Load Radar Params %
radarParams;

javaclasspath('../Libraries/jeromq-0.5.2.jar')
import org.zeromq.*

%subscribe to ZMQ feed
context = ZContext();
socket = context.createSocket(ZMQ.SUB); 
success = false;
while(~success)
    %success = socket.connect('tcp://127.0.0.1:9111');
    %success = socket.connect('tcp://lcrsspdev2:5557');
    success = socket.connect('tcp://192.168.0.10:5557');
end
socket.subscribe("");
socket.setTCPKeepAlive(1);

% Sleep for 10 seconds, give time to connect
pause(10);
isData = true;

figure;
RDMapCh1 = imagesc(nan);
CMapFastCh1 = imagesc(nan);
CMapSlowCh1 = imagesc(nan);
differenceMap = imagesc(nan);
detectionMap = imagesc(nan);

fastCMap = zeros(256,1008);
slowCMap = zeros(256,1008);
differenceMap = zeros(256,1008);
detectionMap = zeros(256,1008);

%%
timeNow = string((datestr(now,'HH-MM-SS-FFF'))); %'HH-MM-SS-FFF'
dateToday = string(date);
dateTime = dateToday + "-" + timeNow; %append(dateToday, timeNow); 
description = "herd.log";
logFileName = dateTime + "-" + description;

logfiledir = ('../../LogFiles'); %laptop
fid = fopen(fullfile(logfiledir, logFileName), 'a');

if fid == -1
    disp("Cannot open log file");
end

%receive a message, expecting a burst at a time
while true
    if isData == true
        message = socket.recv(1); %nonblocking receive uses argument (1)
        fwrite(fid,(string((datestr(now,'HH-MM-SS-FFF')))), 'int8');
        fwrite(fid, message, 'int8');
        rawData = typecast(message, 'uint32');  
        if(length(rawData) < 256*1024) 
            disp("length issue- not a full burst, or No Data ");   
            disp(length(rawData));     
            disp(string((datestr(now,'HH-MM-SS-FFF'))));     
            isData = false;
            fclose(fid);
            close(gcf);
            %exit;
            %socket.close();
            break % think about this carefully
        end
        
        %% Process:
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

        %% FFT's
        fftCh1 = abs(fft2(Ch1MatrixDouble.*Window));        
        fftCh1 = flip(fftCh1,1); % flipping here because we want to see 0 doppler at 0 range
        fftShift = fftshift(fftCh1, 1);
        
        % alpha filter detection
        fastCMap = (1-fastAlpha).*fastCMap + fastAlpha.*fftShift;
        slowCMap = (1-slowAlpha).*slowCMap + slowAlpha.*fftShift;
        
        fastCMapLog = 20*log10(abs(fastCMap));
        slowCMapLog = 20*log10(abs(slowCMap)); 
        diffMap = fastCMapLog - slowCMapLog;
        DetectionMatrix = (DetectionThreshold(diffMap,threshold)); 

        %% Plots
        ax1 = subplot(2,2,1);
        CMapSlowCh1 = imagesc(slowCMapLog(:, 1:maxRangeBin));% , [dBScaleLowerSlow dBScaleUpperSlow]);
        title('Stationary Map')
        xlabel('Range (m)'); ylabel('Doppler (m/s)');
        xticklabels = 0:stepSize:maxRange;
        yticklabels = -4.6:2.3:4.6;
        set(ax1, 'YTick', linspace(1, 256, numel(yticklabels)), 'YTickLabel', flipud(yticklabels(:)), ... 
            'XTick',linspace(1, maxRangeBin, numel(xticklabels)), 'XTickLabel', xticklabels );
        colorbar;

        ax2 = subplot(2,2,2);
        CMapFastCh1 = imagesc(fastCMapLog(:, 1:maxRangeBin)); % , [dBScaleLowerFast dBScaleUpperFast]);
        title('Doppler Map')
        xlabel('Range (m)'); ylabel('Doppler (m/s)');
        xticklabels = 0:stepSize:maxRange;
        yticklabels = -4.6:2.3:4.6;
        set(ax2, 'YTick', linspace(1, 256, numel(yticklabels)), 'YTickLabel', flipud(yticklabels(:)),...
            'XTick',linspace(1, maxRangeBin, numel(xticklabels)), 'XTickLabel', xticklabels );
        colorbar;

        ax3 = subplot(2,2,3);
        differenceMap = imagesc(diffMap(:, 1:maxRangeBin)); %, [dBScaleLowerDiff dBScaleUpperDiff]);
        title('Difference Map')
        xlabel('Range (m)'); ylabel('Doppler (m/s)');
        xticklabels = 0:stepSize:maxRange;
        yticklabels = -4.6:2.3:4.6;
        set(ax3, 'YTick', linspace(1, 256, numel(yticklabels)), 'YTickLabel', flipud(yticklabels(:)),...
            'XTick',linspace(1, maxRangeBin, numel(xticklabels)), 'XTickLabel', xticklabels );
        colorbar;

        ax4 = subplot(2,2,4);
        detectionMap = imagesc(DetectionMatrix(:, 1:maxRangeBin)); %, [dBScaleLowerDet dBScaleUpperDet]);
        title('Detection Map')
        xlabel('Range (m)'); ylabel('Doppler (m/s)');
        xticklabels = 0:stepSize:maxRange;
        yticklabels = -4.6:2.3:4.6;
        set(ax4, 'YTick', linspace(1, 256, numel(yticklabels)), 'YTickLabel', flipud(yticklabels(:)),...
            'XTick',linspace(1, maxRangeBin, numel(xticklabels)), 'XTickLabel', xticklabels );
        colorbar;
        drawnow();
    end
    pause(0.09); %winner is 0.09
end

%when done
socket.close();



    
