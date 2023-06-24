clear all
clc
warning off

% debug (1)
disp("Start:");
disp(datestr(now,'HH:MM:SS'));

% carica dataset
load('skin_dataset.mat','DATA');

% DIM2 = DIM1 + DIM_test_pattern
NF = size(DATA{3},1); % number of folds
DIV = DATA{3}; % divisione fra training e test set
DIM1 = DATA{4}; % numero di training pattern
DIM2 = DATA{5}; % numero di pattern
yE = DATA{2}; % label dei patterns
NX = DATA{1}; % immagini

% carica rete pre-trained
net = alexnet; % load AlexNet
siz = [227 227];

% parametri rete neurale
miniBatchSize = 30;
learningRate = 1e-4;
metodoOptim = 'sgdm';
options = trainingOptions(metodoOptim,...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',30,...
    'InitialLearnRate',learningRate,...
    'Verbose',false,...
    'Plots','training-progress');
numIterationsPerEpoch = floor(DIM1/miniBatchSize);

for fold = 1:NF
    close all force
    
    % debug (2)
    disp("Progress:");
    disp(datestr(now,'HH:MM:SS'));
    disp(fold);
    
    trainPattern = (DIV(fold,1:DIM1(fold))); % training set indexes
    testPattern = (DIV(fold,DIM1(fold)+1:DIM2)); % test set indexes
    y = yE(DIV(fold,1:DIM1(fold))); % training label
    yy = yE(DIV(fold,DIM1(fold)+1:DIM2)); % test label
    numClasses = max(y); % number of classes
    
    % creo il training set
    clear nome trainingImages
    for pattern = 1:DIM1(fold)
        IM = NX{DIV(fold,pattern)}; % singola data immagine
        % si deve fare resize immagini per rendere compatibili con CNN
        IM = imresize(IM,[siz(1) siz(2)]);
        if size(IM,3) == 1
            IM(:,:,2) = IM;
            IM(:,:,3) = IM(:,:,1);
        end
        trainingImages(:,:,:,pattern) = IM;
    end
    imageSize = size(IM);
    
    
    %inserire qui funzione per creare pose aggiuntive, in input si prende
    %(trainingImages,y) e in output restituisci una nuova versione di
    %(trainingImages,y) aggiornata con nuove immagini
    %%%%%%%%%%%
    
    [trainingImages,y] = myImageDataAugmenter(trainingImages,y);
    
    % creazione pattern aggiuntivi mediante tecnica standard
    imageAugmenter = imageDataAugmenter('RandRotation',[-10 10]);
    
    % per testare la rete senza aggiunta di metodi di data augmentation
    %imageAugmenter = imageDataAugmenter('RandRotation',[0 0]);
    
    trainingImages = augmentedImageSource(imageSize,trainingImages,categorical(y'),'DataAugmentation',imageAugmenter);
    
    
    % tuning della rete
    % The last three layers of the pretrained network net are configured for 1000 classes.
    % These three layers must be fine-tuned for the new classification problem. Extract all layers, except the last three, from the pretrained network.
    layersTransfer = net.Layers(1:end-3);
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        softmaxLayer
        classificationLayer];
    netTransfer = trainNetwork(trainingImages,layers,options);
    
    % creo test set
    clear nome test testImages
    % for pattern = ceil(DIM1)+1:ceil(DIM2)
    for pattern = DIM1(fold)+1:DIM2
        IM = NX{DIV(fold,pattern)}; % singola data immagine
        
        IM = imresize(IM,[siz(1) siz(2)]);
        if size(IM,3) == 1
            IM(:,:,2) = IM;
            IM(:,:,3) = IM(:,:,1);
        end
        % testImages(:,:,:,pattern-ceil(DIM1)) = uint8(IM);
        testImages(:,:,:,pattern-DIM1(fold)) = uint8(IM);
    end
    
    % classifico test patterns
    [outclass, score{fold}] = classify(netTransfer,testImages);
    
    % calcolo accuracy
    [a,b] = max(score{fold}');
    ACC(fold) = sum(b==yy)./length(yy)
    
    %salvate quello che vi serve
    %%%%%
    
end

% k-fold mean accuracy
ACC_mean = mean(ACC)

% debug (3)
disp("End:");
disp(datestr(now,'HH:MM:SS'));
