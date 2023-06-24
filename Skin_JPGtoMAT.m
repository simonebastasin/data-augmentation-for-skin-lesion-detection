clear all
clc

% carica label del dataset
load('dataset_label.mat','DATA_label');

% d = dir('C:\Users\simon\Downloads\prog 2\dataverse_files\HAM10000\*.jpg');
% cd('C:\Users\simon\Downloads\prog 2\dataverse_files\HAM10000')
d = dir('C:\Users\simon\Downloads\prog 2\dataverse_files\HAM10000_images_part_1\*.jpg');
cd('C:\Users\simon\Downloads\prog 2\dataverse_files\HAM10000_images_part_1')
clear DATA label DIV

% per ogni immagine
for img = 1:length(d)
    img
    % leggo immagine
    IMG = imread(d(img).name);
    % resize (in questo caso non entra mai)
    while size(IMG,1) > 500
        IMG = imresize(IMG,0.75);
    end
    DATA{1}{img} = uint8(IMG);
    name = extractBetween(convertCharsToStrings(d(img).name),1,12);
    for i = 1:7
        if ismember(name,DATA_label{i}) == 1
            label(img) = i;
        end
    end
end
% 11526 = 10015 + 1511

% label dei patterns
DATA{2} = label;

% 5-fold cross-validations
indices = crossvalind('Kfold',label,5);
for fold = 1:5
    test = (indices == fold);
    train = ~test;
    % divisione fra training e test set
    DATA{3}(fold,:) = [find(train); find(test)];
    % numero di training pattern
    DATA{4}(fold) = length(find(train));
end
% numero di pattern
DATA{5}=length(DATA{1});

% test = zeros(11526,1);
% test(10015+1:11526) = ones(11526-10015,1);
% train = ~test;

% salvataggio dataset in .mat
save('skin_subdataset_4.mat','DATA','-v7.3');

ok = 1