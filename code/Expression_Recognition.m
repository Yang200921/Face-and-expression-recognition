% To run this program on your own computer, the areas that need to be modified include:
% Ensure that all required MATLAB toolboxes, including the deepNetworkDesigner, are installed, along with the MobileNetV2 neural network model.
% Line 16, faceDatasetPath should be set to the dataset path in your computer.
% Line 58, if there is no GPU available, change the setting to CPU.
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
FaceDataPath = fullfile('/Users/yangsansui/Desktop/project/Dataset/Expression');  
imds = imageDatastore(FaceDataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
proportionFaceTrain = 0.8; % 80% for training
proportionFaceValidation = 0.1; % 10% for validation
[imds_train, imds_test, imds_valid] = splitEachLabel(imds, proportionFaceTrain, proportionFaceValidation, 'randomized'); % 10% for testing

% Data enhancement through data preprocessing including reflection, translation, rotation and uniformity of size and channel count.
pixelRange = [-5 5];    
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandRotation', [-15, 15]);

augimds_train = augmentedImageDatastore(inputSize(1:2), imds_train,'DataAugmentation',imageAugmenter,"ColorPreprocessing","gray2rgb");
augimds_test = augmentedImageDatastore(inputSize(1:2), imds_test,'DataAugmentation',imageAugmenter,"ColorPreprocessing","gray2rgb");
augimds_valid = augmentedImageDatastore(inputSize(1:2), imds_valid,'DataAugmentation',imageAugmenter,"ColorPreprocessing","gray2rgb");
numClasses = numel(categories(imds_train.Labels));

% Adjust the structure of neural network
% Modify layers with the number of classifications
lgraph = removeLayers(lgraph, {'Logits','Logits_softmax','ClassificationLayer_Logits'});
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc_func','Weights',rand(numClasses,1280),'Bias', ones(numClasses,1), 'WeightL2Factor',0.005)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','new_classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'global_average_pooling2d_1','fc_func');

% If the trained model activation function is ReLu6, delete this section.
% Replace all ReLU6 layers with Leaky ReLU layers
for i = 1:length(lgraph.Layers)
    layer = lgraph.Layers(i);
    if contains(layer.Name, 'relu')
        leakyLayer = leakyReluLayer( 0.01, 'Name', [layer.Name '_leaky']);
        lgraph = replaceLayer(lgraph, layer.Name, leakyLayer);
    end
end

% Set training parameters
net = assembleNetwork(lgraph);
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'gpu', ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 10, ...
    'MiniBatchSize', 32, ...
    'ValidationData',augimds_valid, ...
    'ValidationFrequency',100, ...
    'Shuffle','every-epoch', ...
    'MaxEpochs', 60, ...
    'Verbose', true, ...
    'Plots','training-progress');

% Training and saving the neural network model
model = trainNetwork(augimds_train,lgraph,options);
save('network/MobileNet_expressionRecognition.mat', 'model');

%%
% Data visualization, randomly show two pictures in each category
[subsetImds, ~] = splitEachLabel(imds_train, 2, 2, 'randomized');
figure('name','Random Two Samples from Each Category')
montage(subsetImds.Files, 'Size', [4 4]);
title('Random Two Samples from Each Category')

% CAM visualization
% Showed the random two input images by adjusting the number of channels and size.
numImages = numel(imds.Files);
idx = randperm(numImages, 2);
img1 = readimage(imds, idx(1));
img1 = imresize(img1,[224 224]);
img1 = cat(3, img1, img1, img1);
img2 = readimage(imds, idx(2));
img2 = imresize(img2,[224 224]);
img2 = cat(3, img2, img2, img2);
figure('name','Input images')
subplot(1, 2, 1);
imshow(img1);
subplot(1, 2, 2);
imshow(img2);
images = {img1, img2};

% Separately obtain the feature heatmaps obtained by two input images in classification layer and resize to show them.
classes = net.Layers(end).Classes;
for i = 1:numel(images)
    % Get activations
    currentImg = images{i};
    currentImg = imresize(currentImg,[inputSize(1), NaN]);
    imageActivations = activations(net, currentImg, 'out_relu_leaky');

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
YPred = classify(model, augimds_test);
YTrue = imds_test.Labels;
accuracy = mean(YPred == YTrue)

% Show the Confusion Matrix
YPredIdx = grp2idx(YPred);
YTrueIdx = grp2idx(YTrue);
cm = confusionmat(YTrueIdx, YPredIdx);
figure
confusionchart(cm, unique(imds.Labels), 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Confusion Matrix for Classification');

%%
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


