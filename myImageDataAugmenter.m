function [trainImages,trainLabels] = myImageDataAugmenter(trainImages,trainLabels)

% number of augmented images of each original one
n = 9;

% starting position of new images
k = length(trainImages(1,1,1,:));

% for image warping
R = makeresampler({'cubic','nearest'},'replicate');
r = @(x) sqrt(x(:,1).^2 + x(:,2).^2);
w = @(x) atan2(x(:,2), x(:,1));

% for each image in training set
for i = 1:length(trainImages(1,1,1,:))
    
    % for each of 9 new images
    for j = 1:n
        clear augImg
        augImg(:,:,:,j) = trainImages(:,:,:,i);
        
        %% 1) Flips
        if randi(2) == 1 % 50% chance
            % horizontal flip
            augImg(:,:,:,j) = flipdim(augImg(:,:,:,j),2);
        end
        if randi(2) == 1  % 50% chance
            % vertical flip
            augImg(:,:,:,j) = flipdim(augImg(:,:,:,j),1);
        end
        
        %% 2) Rotation
        % angle = 90*randi([0,3]) = {0; 90; 180; 270} (in degrees)
        augImg(:,:,:,j) = imrotate(augImg(:,:,:,j),90*randi([0,3]));
        
        %% 4) Shear
        tanMaxAngle = 0.0607; % tan(20) = 0.0607
        shearOffset = 6.889; % tan(20)*227/2 = 6.889 with imgSize=[227 227]
        shearCase = randi([-6,6]);
        a = tanMaxAngle*shearCase;
        shearOffset = shearOffset*shearCase;
        if randi(2) == 1 % 50% chance
            % horizontal shear
            T = maketform('affine',[1 0 0; a 1 0; 0 0 1]);
            augImg(:,:,:,j) = imtransform(augImg(:,:,:,j),T,R,'XData',[1+shearOffset 227+shearOffset],'YData',[1 227]);
        else % 50% chance
            % vertical shear
            T = maketform('affine',[1 a 0; 0 1 0; 0 0 1]);
            augImg(:,:,:,j) = imtransform(augImg(:,:,:,j),T,R,'XData',[1 227],'YData',[1+shearOffset 227+shearOffset]);
        end
        
        %% 3) Crops
        %if randi(2) == 1 % 50% chance
            % keep from 40% to 100% of the original image
            rect = randomWindow2d([227 227],'Scale',[0.4 1],'DimensionRatio',[1 1; 1 1]);
            helpImg = imcrop(augImg(:,:,:,j),rect);
            augImg(:,:,:,j) = imresize(helpImg,[227 227]);
        %end
        
        %% 5) Saturation, Contrast, Brightness
        if randi(2) == 1 % 50% chance
            % alteration of image contrast, saturation and brightness
            augImg(:,:,:,j) = jitterColorHSV(augImg(:,:,:,j),'Contrast',0.2,'Saturation',0.2,'Brightness',0.2);
        %end
        %{
        %% 6) Hue
        %if randi(2) == 1 % 50% chance
            % alteration of image hue
             augImg(:,:,:,j) = jitterColorHSV(augImg(:,:,:,j),'Hue',0.1);
        %end
        
        %% 7) Spatial Transformations
        % starting as shrinkage
        a = randi([9 10])/10;
        if randi(2) == 1 % 50% chance
            % change to expansion
            a = 2-a;
        end
        f = @(x) [r(x).^a .* cos(w(x)), r(x).^a .* sin(w(x))];
        g = @(x, unused) f(x);
        tform = maketform('custom',2,2,[],g,[]);
        augImg(:,:,:,j) = imtransform(augImg(:,:,:,j),tform,R,'UData',[-1 1],'VData',[-1 1],'XData',[-1 1],'YData',[-1 1]);
        %}
        %% Add the new image to training set
        trainImages(:,:,:,k+j) = augImg(:,:,:,j);
        trainLabels(k+j) = trainLabels(i);
        
    end
    
    k = k + j; % next starting position for next new images
end

end