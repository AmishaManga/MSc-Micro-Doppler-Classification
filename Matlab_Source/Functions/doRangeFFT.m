function rangeFFT = doRangeFFT(rawDataMatrixUint32, noOfRangeBins, noOfDopplerBins)

    rawDataVectorUint32 = reshape(rawDataMatrixUint32', [], 1);

    rawDataVectorInt16 = typecast(rawDataVectorUint32, 'int16');

    rawDataMatrixInt16 = reshape(rawDataVectorInt16, noOfRangeBins, noOfDopplerBins)';
    
    rawDataMatrixDouble = double(rawDataMatrixInt16);
    
    rawDataMatrixDouble = rawDataMatrixDouble - repmat(mean(rawDataMatrixDouble, 1), size(rawDataMatrixDouble, 1), 1);
    rawDataMatrixDouble = rawDataMatrixDouble - mean(mean(rawDataMatrixDouble));
    
    RWindow = repmat(Hann(noOfRangeBins), noOfDopplerBins, 1);
    RWindow = RWindow / sum(RWindow);

    nextPowOf2Range = 2^(nextpow2(noOfRangeBins));

    rangeFFT = fft(rawDataMatrixDouble.*RWindow, nextPowOf2Range,2);

end