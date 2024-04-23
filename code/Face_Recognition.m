% To run this program on your own computer, the areas that need to be modified include:
% Ensure that all required MATLAB toolboxes, including the deepNetworkDesigner, are installed, along with the MobileNetV2 neural network model.
% Line 16, faceDatasetPath should be set to the dataset path in your computer.
% Line 48, if there is no GPU available, change the setting to CPU.
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Neural network %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the neural network
net = mobilenetv2;
lgraph = layerGraph(net);
inputLayer = net.Layers(1);
inputSize = inputLayer.InputSize;

% Seperate training set, validation set and test set.
FaceDataPath = fullfile('/Users/yangsansui/Desktop/project/Dataset/FaceDataset');  
imds_faces = imageDatastore(FaceDataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
numClasses = numel(categories(imds_faces.Labels));
numFaceTrain = 7; % 70% for training
numFaceValidation = 1; % 10% for validation
[imds_FaceTrain, imds_FaceValidation, imds_FaceTest] = splitEachLabel(imds_faces, numFaceTrain, numFaceValidation, 'randomized'); % 20% for testing

% Data enhancement through data preprocessing including reflection, translation, rotation and uniformity of size and channel count.
pixelRange = [-10 10];    
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandRotation', [-15, 15]);
augimdsFaceTrain = augmentedImageDatastore(inputSize(1:2),imds_FaceTrain,'DataAugmentation',imageAugmenter, 'ColorPreprocessing','gray2rgb');
augimdsFaceValidation = augmentedImageDatastore(inputSize(1:2),imds_FaceValidation,'DataAugmentation',imageAugmenter, 'ColorPreprocessing','gray2rgb');
augimdsFaceTest = augmentedImageDatastore(inputSize(1:2),imds_FaceTest,'DataAugmentation',imageAugmenter, 'ColorPreprocessing','gray2rgb');

% Adjust the structure of neural network
lgraph = removeLayers(lgraph, {'Logits','Logits_softmax','ClassificationLayer_Logits'});
newLayers = [
    dropoutLayer(0.4,'Name','New_Dropout') 
    fullyConnectedLayer(numClasses,'Name','fc_func', 'Weights',rand(numClasses,1280),'Bias', ones(numClasses,1), 'WeightL2Factor',0.005)    
    % The value of Weights depends on the output of last layer
    softmaxLayer('Name','softmax')
    classificationLayer('Name','new_classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'global_average_pooling2d_1','New_Dropout');
net = assembleNetwork(lgraph);

% Set training parameters
options = trainingOptions('adam', ...   
    'ExecutionEnvironment', 'gpu', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 10, ...
    'MiniBatchSize', 8, ...
    'ValidationData', augimdsFaceValidation, ...
    'ValidationFrequency',30, ...
    'Shuffle','every-epoch', ...
    'MaxEpochs', 30, ...
    'Verbose', true, ...
    'Plots','training-progress');

% Training and saving the neural network model
model = trainNetwork(augimdsFaceTrain,lgraph,options);
save('networl/MobileNet_faceRecognition.mat', 'model');

%%
% Data visualization, randomly show 16 images
numTrainImages = numel(imds_FaceTrain.Labels);
idx = randperm(numTrainImages,16);
figure('name','Random Images Display')
for i = 1:16
    subplot(4,4,i)
    I = readimage(imds_FaceTrain,idx(i));
    imshow(I)
end

% Feature map visualization
% Showed the random two input images by adjusting the number of channels and size.
idx = randperm(numTrainImages,2);
img1 = readimage(imds_FaceTrain, idx(1));
img1 = imresize(img1,[224 224]);
img1 = cat(3, img1, img1, img1);
img2 = readimage(imds_FaceTrain, idx(2));
img2 = imresize(img2,[224 224]);
img2 = cat(3, img2, img2, img2);
figure('name','Input images')
subplot(1, 2, 1);
imshow(img1);
subplot(1, 2, 2);
imshow(img2);

% Separately obtain the feature maps obtained by two input images in first and intermediate convolution layers and resize to show them.
images = {img1, img2};
layers = {'Conv1', 'block_7_project'};
for i = 1:numel(images)
    img = images{i};
    for j = 1:numel(layers)
        layer = layers{j};
        act = activations(net, img, layer);
        sz = size(act);
        act = reshape(act, [sz(1), sz(2), sz(3)]);
        if j == 2
            I = imtile(mat2gray(act), 'Gridsize', [8 8]);
            I = imresize(I, 6);
            figure('name', ['Image ', num2str(i), ' - ', layer, ' Feature']);
        else
            I = imtile(mat2gray(act), 'Gridsize', [4 8]);
            figure('name', ['Image ', num2str(i), ' - ', layer, ' Feature']);
        end
        imshow(I);
    end
end

% CAM visualization
% Separately obtain the feature heatmaps obtained by two input images in classification layer and resize to show them.
net = assembleNetwork(layerGraph(model));
classes = net.Layers(end).Classes;
for i = 1:numel(images)
    % Get activations
    currentImg = images{i};
    currentImg = imresize(currentImg,[inputSize(1), NaN]);
    imageActivations = activations(net, currentImg, 'out_relu');

    % Calculate scores
    scores = squeeze(mean(imageActivations,[1 2]));
    fcWeights = net.Layers(end-2).Weights;
    fcBias = net.Layers(end-2).Bias;
    scores =  fcWeights * scores + fcBias;

    % Get top three classes
    [~, classIds] = maxk(scores, 3);

    % Get weight vector and compute class activation map
    weightVector = shiftdim(fcWeights(classIds(1), :), -1);
    classActivationMap = sum(imageActivations .* weightVector, 3);

    % Compute max scores and class labels
    scores = exp(scores) / sum(exp(scores));     
    maxScores = scores(classIds);
    labels = classes(classIds);

    % Display image and class activation map
    figure('name','heat features')
    subplot(1,2,1)
    imshow(currentImg)
    subplot(1,2,2)
    CAMshow(currentImg, classActivationMap)
    title(string(labels) + ", " + string(maxScores));
end

%%
% Performance Evaluation
% Compute accuracy
YPred = classify(model, augimdsFaceTest);
YTrue = imds_FaceTest.Labels;
accuracy = mean(YPred == YTrue)

% Show the Confusion Matrix
YPredIdx = grp2idx(YPred);
YTrueIdx = grp2idx(YTrue);
cm = confusionmat(YTrueIdx, YPredIdx);
figure
confusionchart(cm, unique(imds_FaceTest.Labels), 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Confusion Matrix for Classification');

% Nested functions used by CAM calculation
function CAMshow(im,CAM)
imSize = size(im);
CAM = imresize(CAM,imSize(1:2));
CAM = normalizeImage(CAM);
CAM(CAM<0.2) = 0;
cmap = jet(255).*linspace(0,1,255)';
CAM = ind2rgb(uint8(CAM*255),cmap)*255;

combinedImage = double(rgb2gray(im))/2 + CAM;
combinedImage = normalizeImage(combinedImage)*255;
imshow(uint8(combinedImage));
end

function N = normalizeImage(I)
minimum = min(I(:));
maximum = max(I(:));
N = (I-minimum)/(maximum-minimum);
end