% MatrixOut = DetectionThreshold(MatrixIn, threshold)
%

function MatrixOut = DetectionThreshold(MatrixIn, threshold)
    
    for i = 1:256
        for j = 1:1008
            if MatrixIn(i,j) <= threshold
                MatrixOut(i,j) = 0;
            else
                MatrixOut(i,j) = 1;
            end
        end        
    end

end