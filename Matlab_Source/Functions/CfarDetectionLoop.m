% thresholdMatrix = CfarDetectionLoop(MatrixIn,Pfa,N,nGuardCell)
% This function passes Row by Row, that is each range in the R-D into the Cfar function

function MatrixOut = CfarDetectionLoop(MatrixIn,Pfa,nReferenceCells,nGuardCell)
    
    for i = 1:256
        Row = MatrixIn(i,:);
        [T1]= CA_CFAR(Pfa,nReferenceCells,nGuardCell ,Row); 
        MatrixOut(i,:) = T1';
    end
end