% MatrixOut = DetectionThreshold(MatrixIn, threshold)
%

function MatrixOut = SpecgramThreshold(MatrixIn, threshold)
    
    for i = 1:1024
        for j = 1:464
            if MatrixIn(i,j) <= threshold
                MatrixOut(i,j) = 0;
            else
                MatrixOut(i,j) = 1;
            end
        end        
    end

end