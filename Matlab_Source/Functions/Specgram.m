function [spectrogramMatrix] = Specgram(signal,windowLength,overlap, nfft)
% Computes the Short-Time Fourier Transform (STFT) of an input signal, producing a spectrogram.

% signal : Input Signal
% windowLength: The length of each segment for analysis, in samples.
% overlap : The number of samples to overlap between segments.
% nfft: The number of points in the Discrete Fourier Transform (DFT).

% Returns:
% spectrogramMatrix: The STFT of the signal, represented as a matrix. Each column corresponds to a time frame,
%                     while each row represents a frequency bin.

    % Determine the total length of the input signal
    signalLength = length(signal);

    % Initialize the STFT result as an empty matrix
    STFT = [];
    
    % Calculate the total number of segments to analyze, accounting for overlap
    if (overlap == 0)
        totalSegments = floor(signalLength/windowLength);
    else 
        segmentsFromOverlap = floor(signalLength/overlap);
        totalSegments = segmentsFromOverlap  - floor(windowLength/overlap);
    end
    
    % Loop through each segment to compute its Fourier Transform
    for segIndex = 1: totalSegments
        % Calculate the start and end indices of the current segment
        segStart = (segIndex-1)*overlap + 1;
        segEnd = segStart + windowLength -1;

        % Extract the segment of the signal
        fftSeg = signal(segStart:segEnd);

        % Apply the window function to the segment
        windowedSegment = Hann(length(fftSeg))';
        windowedSegment = windowedSegment / sum(windowedSegment);

        % Compute the Fourier Transform of the windowed segment and center the zero frequency
        STFT(:,segIndex) = fftshift(fft(fftSeg.*windowedSegment, nfft)); 
    end
    % Assign the computed STFT to the output variable
	spectrogramMatrix = STFT;
end