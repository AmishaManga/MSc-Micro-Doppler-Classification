% thresholdMatrix = CfarDetectionLoop(MatrixIn,Pfa,N,nGuardCell)
% This function passes Row by Row, that is each range in the R-D into the Cfar function

function thresholdMatrix = CfarDetectionLoopDoppler(RangeDopplerMap,Pfa,nReferenceCells,nGuardCell)
    
    for i = 1:1008
        Column = RangeDopplerMap(:,i);
        [T1]=CA_CFAR(Pfa,nReferenceCells,nGuardCell, Column');  
        thresholdMatrix(:,i) = T1;
    end
end