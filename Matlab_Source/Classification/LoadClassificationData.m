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

%% Load Radar Params and Range Bins
radarParams;

datapath = 'E:ReadyToClassifyFinal\AugmentedSets\Hann\Hann_PintoCH1.mat'

%% Load Mat Files
ClassificationTestData = load(datapath);
[d1,indexOfMatFile] = size(ClassificationTestData.X);

numSpecgrams = indexOfMatFile;
    
X = 20*log10(abs(ClassificationTestData.X));
Y = ClassificationTestData.Y;
Yt = Y';

for i = 1:numSpecgrams 
    data(i, :) = X(:)'  ;
    
    %% Draw Spectrogram
    SpecData = ClassificationTestData.X(i,:); 
    reshapedMatrix = reshape(SpecData, 624, 256)';
    burstCount = 20;
    dbdown = 80;
    nfft = 256; 

    SpecData = imagesc(reshapedMatrix);
    set(gca,'CLim', [-30 -10]);
    colorbar;
    colormap(jet);
    title("Spectrogram");
    ylabel('Velocity(m/s)'); 
    xlabel('Time (s)'); 
    colormap(jet);
    yticklabels = -4.6:2.3:4.6; 
    xticklabels = 0:1:7; 
    set(gca, 'YTick', linspace(1, 256, 5), 'YTickLabel', flipud(yticklabels(:)), ...
    'XTick',linspace(1, 624, 8), 'XTickLabel', xticklabels );
    pause;
    
    i = i+1;
end
    
 