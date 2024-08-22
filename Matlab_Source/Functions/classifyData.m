function button = classifyData(data) 
filename = 'C:\Users\Amisha\Documents\LinkCh1Dog.mat';

if exist(filename, 'file') == 2
    ClassificationTestData = load(filename);
    [d1,indexOfMatFile] = size(ClassificationTestData.X); 
    [d1,indexOfMatFile] = size(ClassificationTestData.Y);
    
    i = 1 + indexOfMatFile;
    X = ClassificationTestData.X;
    Y = ClassificationTestData.Y;
else 
    i = 1;
end

button = MFquestdlg ( [ 0.6 , 0.1 ] , 'What do you see?'...
    , 'Classification Menu', ...
    'Human', 'Animal', 'Noise', 'Noise');

% Handle response
switch button
    case 'Human'
        dataMatrixObject1 = data;
        [d1,d2] = size(dataMatrixObject1);
        X(i,:) = reshape(dataMatrixObject1',1,d1*d2);   
        Y(i) = 1;
        i = i+1;
        save(filename, 'X', 'Y');     
        return;

    case 'Animal'
        dataMatrixObject1 = data;
        [d1,d2] = size(dataMatrixObject1);
        X(i,:) = reshape(dataMatrixObject1',1,d1*d2);   
        Y(i) = 2; %change this 1 for human 2 for animal
        i = i+1;
        save(filename, 'X', 'Y');   
        return
        
    case 'Noise'
        disp('Do Nothing - Noise');
        return

        
end

end
