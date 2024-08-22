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

%% Load Mat Files
ClassificationTestData = load('C:\Users\Amisha\Documents\Reutech1.mat');
% comp = pca(ClassificationTestData.X);

index = 2;
    
%% Draw Spectrogram
SpecData = ClassificationTestData.X(index,:); 
reshapedMatrix = reshape(SpecData, 624, 256)';

SpecData = imagesc(reshapedMatrix);
set(gca,'CLim', [-30 -10]);
colorbar;
title("Spectrogram");
ylabel('Velocity(m/s)'); 
xlabel('Time (s)'); 
colormap(jet);
% yticklabels = -4.6:2.3:4.6; 
% xticklabels = 0:1:7; 
% set(gca, 'YTick', linspace(1, 256, 5), 'YTickLabel', flipud(yticklabels(:)), ...
% 'XTick',linspace(1, 624, 8), 'XTickLabel', xticklabels );
    

    
 