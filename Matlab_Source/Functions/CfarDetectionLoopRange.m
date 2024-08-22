% thresholdMatrix = CfarDetectionLoop(MatrixIn,Pfa,N,nGuardCell)
% This function passes Row by Row, that is each range in the R-D into the Cfar function

function thresholdMatrix = CfarDetectionLoopRange(RangeDopplerMap,Pfa,nReferenceCells,nGuardCell)
    
    for i = 1:256
        Row = RangeDopplerMap(i,:);
        [T1]= CA_CFAR(Pfa,nReferenceCells,nGuardCell ,Row); 
        thresholdMatrix(i,:) = T1';
    end
end