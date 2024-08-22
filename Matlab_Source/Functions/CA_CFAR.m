% [T,iCFARwin]=CA_CFAR(Pfa,N,nGuardCell, signal)
%
% [T,iCFARwin]=CA_CFAR(Pfa,N,nGuardCell, signal) produces a vector T,
% that is the threshold produced by the Cell-Averaging CFAR (CACFAR)
% method from the vector, signal, that is a one-dimension of range
% power.

% INPUT:
% Pfa - the desired probability of False Alarm
% N - the total number of CFAR windows-- with N/2 in the leading window
% and N/2 in the lagging window.
% nGuardCell- the number of guard cells to either side of the test cell
% Signal - the power of the recieved waveform after having passed through
% a square law detector. It is upon this that the threshold is set.
% OUTPUT:
% T is the threshold set according to the functional inputs.

% AUTHORSHIP:
% James Jen
% Cal Poly Pomona

function [T]=CA_CFAR(Pfa,N,nGuardCell,signal)

    %Threshold multiplier in power
    cFarThreshold= N*(Pfa^(-1/N)-1);
    
    %Number of Range Bins
    L=length(signal);
   
    % create window index using guard cells and reference cells
    iWindow=1:N+1+2*nGuardCell; 
    nCUT=L-N-2*nGuardCell; 
    shiftCFAR=(0:nCUT-1).'; 
    index=repmat(iWindow,nCUT,1)+repmat(shiftCFAR,1,N+1+2*nGuardCell);

    iRefWin=index;
    i=N/2+(1:2*nGuardCell+1);
    iRefWin(:,i)=[];

    i=N/2+nGuardCell+1;
    iCUT=index(:,i);
    
    %Summing the content of each reference window
    B=(sum(signal(iRefWin),2)/N).^2;
    
    %For start/end of vector that is not included in window as it doesnt make up a full window for the algorithm: 
    %Use one side of the window and slide along till vectorz is no longer empty
    halfWindow = i-1;
    startB = zeros(halfWindow,1);
    endB = zeros(halfWindow,1);
    for j= 1:(halfWindow)
        startA = signal(j:(halfWindow+j-1));
        endA = signal(end-halfWindow-j+2:end-j+1);
        
        startAvA = sum(startA)/halfWindow;
        endAvA = sum(endA)/halfWindow;
        
        startB(j) = startAvA.^2;
        endB(end-j+1) = endAvA.^2;
    end
    
    % Concatenate the first half window to the summed up Signal Averages:
    addStartHalfWindow = cat(1,startB, B);
    
    % Concatenate the end half window to the summed up Signal Averages:
    FinalB = cat(1, addStartHalfWindow, endB);
    
    %Outputted threshold
    %T=zeros(L,1); %For CUT too small or too big for complete
    %Reference window, we set T to zero
    
    %Multiplying threshold multiplier
    T=cFarThreshold*FinalB; 
end




