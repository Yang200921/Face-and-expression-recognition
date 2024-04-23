function EnrollFace    % enroll new resident's face and be trained by network to recognize new resident
%%
% To run this program on your own computer, the areas that need to be modified include:
% Ensure that all required MATLAB toolboxes, including the deepNetworkDesigner and Computer Vision Toolbox, are installed, along with the MobileNetV2 neural network model.
% Line 15, videoinput() should adapt to your device.
% Line 109 and 284, faceDatasetPath should be set to the dataset path in your computer.
% Line 306, if there is no GPU available, change the setting to CPU.
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Open Camera %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
camera = [];
vid = [];
if isempty(camera)
    vid = videoinput('macvideo',1,'YCbCr422_1280x720'); 
    set(vid, 'ReturnedColorSpace', 'rgb');           % Set color space to RGB
    vidRes = get(vid, 'VideoResolution'); 
    width = vidRes(1); 
    height = vidRes(2);
    nBands = get(vid,'NumberOfBands'); 
    hImage = image(zeros(height, width, nBands));    % Create an empty image with fixed size
    camera = preview(vid,hImage);                    % Preview the camera
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Build a GUI %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fig = figure('Position',[100,150,980,500]);

Pnl1 = uipanel(Fig,'Position',[0.05,0.05,0.35,0.8]);   
Pnl2 = uipanel(Fig,'Position',[0.425,0.05,0.35,0.8]);   
Pnl3 = uipanel(Fig,'Position',[0.8,0.05,0.15,0.8]);   
Pnl4 = uipanel(Fig,'Position',[0.05,0.875,0.9,0.1]); 

Axes1 = axes(Pnl1,'Position',[0,0.3,1,0.55]);
Axes2 = axes(Pnl2,'Position',[0,0.3,1,0.55]);
axis off;

Bt1 = uicontrol(Pnl3,'style','PushButton','String','Undo','Fontsize',14,'Units','normalized','Position',[0.1,0.8,0.8,0.05],'Callback',@Undo);
Bt2 = uicontrol(Pnl3,'style','PushButton','String','Snap','Fontsize',14,'Units','normalized','Position',[0.1,0.65,0.8,0.05],'Callback',@Snap);
Bt3 = uicontrol(Pnl3,'style','PushButton','String','Submit','Fontsize',14,'Units','normalized','Position',[0.1,0.5,0.8,0.05],'Callback',@Submit);
Bt4 = uicontrol(Pnl3,'style','PushButton','String','Reset','Fontsize',14,'Units','normalized','Position',[0.1,0.35,0.8,0.05],'Callback',@Reset);

lb1 = uicontrol(Pnl1,'style','text','String','Live','Fontsize',17,'Units','normalized','Position',[0.35,0.875,0.3,0.05]);
lb2 = uicontrol(Pnl2,'style','text','String','Shot','Fontsize',17,'Units','normalized','Position',[0.35,0.875,0.3,0.05]);
lb3 = uicontrol(Pnl4,'style','text','String','Face Enrolling System','Fontsize',20,'Units','normalized','Position',[0.15,0.15,0.7,0.7]);

Cb = uicontrol(Pnl2,'style','checkbox','Units','normalized','Position',[0.95,0.1625,0.05,0.05],'Callback',@CreateFolder,'Tag', 'Cb');

Inputbox = uicontrol(Pnl2,'style','edit','String','Enter Name Here','Fontsize',14,'Units','normalized','Position',[0.01,0.15,0.94,0.075],'Tag','Inputbox');
Output_instruction = uicontrol(Pnl1,'style','edit','String','Positive Face','Fontsize',14,'Units','normalized','Position',[0.15,0.15,0.7,0.075],'Tag','Output_instruction');
Output_complete = uicontrol(Pnl3,'style','edit','String',' ','Fontsize',14,'Units','normalized','Position',[0.1,0.2,0.8,0.075],'Tag','Output_complete');

drawnow                                              % Draw the GUI with set components
    
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FolderPath = '';                                     % the path of new user's folder
faceDetector = vision.CascadeObjectDetector();       % Create a cascade object detector for detecting positive face
gap = 3;                                             % Speed up the calculation
snapCounter = 0;
FolderName = '';
accuracy = 0;

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while ishandle(camera)
    frame = getsnapshot(vid);
    frame = im2double(frame);
    imshow(frame,'Parent',Axes1)                     % Display the frame in Live Axes

    % When multiple faces are detected, all buttons are disabled. When only one face detected, frame the recognized face in Live.
    if ~isempty(FolderPath)                          % Face detection is not performed until a new user folder is created.
        bbox = facedetect(frame);                    % Detect faces in the frame by customized function
        num_faces = size(bbox, 1);                   % Iterate through each detected face and draw a rectangle around it。
        for i = 1:num_faces       
            rc = bbox(i, :) + [-bbox(i, 3)/4, -bbox(i, 4)/4, bbox(i, 3)/2, bbox(i, 4)/2];
            rectangle('Position', rc, 'Curvature', 0, 'LineWidth', 2, 'LineStyle', '--', 'EdgeColor', 'y', 'Parent', Axes1);
        end 
    end
    pause(0.1);                                      % Pause 0.1s for stabilize system load
end

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Nested Functions %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function bbox = facedetect(frame)                % detect the faces and record their positions
        frame = frame(1:gap:end,1:gap:end,:);
        bbox = step(faceDetector,frame);
        bbox = bbox*gap;
    end

    function box = restrict(box)                     % restrict the face box within the frame
        box = box+[-box(3)/4,-box(4)/4,box(3)/2,box(4)/2];
        box(1) = max(box(1), 1);
        box(2) = max(box(2), 1);
        box(3) = min(box(3), size(frame, 2) - box(1));
        box(4) = min(box(4), size(frame, 1) - box(2));
    end


    function CreateFolder(~, ~)                      % Create a folder with the name entered by the user.
        FaceDatasetPath = '/Users/yangsansui/Desktop/project/Dataset/FaceDataset';
        Inputbox = findobj(gcf, 'Tag', 'Inputbox');  % Find objects in the graphics object hierarchy to ensure they can be found
        Cb = findobj(gcf, 'Tag', 'Cb');              
        Output_complete = findobj(gcf, 'Tag', 'Output_complete');
        FolderName = get(Inputbox, 'String');
        FolderPath = fullfile(FaceDatasetPath, FolderName);
        if exist(FolderPath, 'dir')
            set(Inputbox, 'String', 'Name already taken. Please choose another.');
            set(Cb, 'Value', 0);
        else
            mkdir(FolderPath);
            set(Inputbox, 'String', 'Folder created.');
            set(Output_complete, 'String', FolderName);
        end
    end

    function snapCounter = Switch(snapCounter)       % switch the instructions while the snap is clicked
        switch snapCounter
            case 0
                set(Output_instruction, 'String', 'Positive face');
            case 1
                set(Output_instruction, 'String', 'Slightly left');
            case 2
                set(Output_instruction, 'String', 'Slightly right');
            case 3
                set(Output_instruction, 'String', 'Slightly upwards');
            case 4
                set(Output_instruction, 'String', 'Slightly downwards');
            case 5
                set(Output_instruction, 'String', 'Finish');
            otherwise
                snapCounter = 0;
                set(Output_instruction, 'String', '');
        end
    end

    function Snap(~, ~)                              % Take two photos for one direction and save them in created folder
        Inputbox = findobj(gcf, 'Tag', 'Inputbox');
        Output_instruction = findobj(gcf, 'Tag', 'Output_instruction');

        if ~isempty(FolderPath)                      % Only happen when folder is not empty.
            snapCounter = snapCounter + 1;
            snapCounter = Switch(snapCounter);
            set(Inputbox, 'String', '');
        else
            set(Inputbox, 'String', 'Please create a folder first.');
            return;
        end

        % After collecting the images, prompt the user to click 'Submit'
        if snapCounter == 6
            snapCounter = 0;
            set(Inputbox, 'String', 'Please click Submit.');
            return
        end

        % Multiple faces in the frame before and during shooting, or movement of the face causing recognition failure, will require a reshoot for this direction.
        for i = 1:2
            photo = getsnapshot(vid);
            photo = im2double(photo);
            bbox = step(faceDetector,photo);
            if i == 1 && size(bbox, 1) ~= 1          % Multiple faces
                set(Inputbox, 'String', 'Ensure only one person in the frame.');
                snapCounter = snapCounter - 1;
                snapCounter = Switch(snapCounter);
                return
            else
                if i == 2 && size(bbox, 1) ~= 1      % movement of the face causing recognition failure
                    files = dir(fullfile(FolderPath, '*.jpg'));
                    delete(fullfile(FolderPath, files(end).name));
                    set(Inputbox, 'String', 'Please keep stable.');
                    snapCounter = snapCounter - 1;
                    snapCounter = Switch(snapCounter);
                    cla(Axes2);
                    return
                end
                % Adjust the sizes of the saved images to facilitate training with the neural network.
                location = restrict(bbox(1, :));
                photo = imcrop(photo, location);
                photo = rgb2gray(photo);
                h_photo = imresize(photo, [224 NaN], 'Method', 'bilinear');
                [~, w] = size(h_photo);
                if w <= 224
                    padding = floor((224 - w) / 2);
                    photo = padarray(h_photo, [0 padding], 0, 'both');
                else
                    w_photo = imresize(photo, [NaN 224], 'Method', 'bicubic');
                    [h, ~] = size(w_photo);
                    padding = floor((224 - h) / 2);
                    photo = padarray(w_photo, [padding 0], 0, 'both');
                end
                % Save images with time stamp to distinguish different images
                timeStamp = datetime("now");
                fileName = sprintf('%s.jpg', timeStamp);
                filePath = fullfile(FolderPath, fileName);
                imwrite(photo,filePath);
                imshow(photo, 'Parent', Axes2);
                pause(1)
            end
        end
    end

    function Undo(~, ~)                              % Undo the last snap function.
        if isempty(FolderPath)                       % Only happen when folder is not empty.
            Inputbox = findobj(gcf, 'Tag', 'Inputbox');
            set(Inputbox, 'String', 'Please create a folder first.');
            return;
        end
        
        cla(Axes2);                                  % Clear the display of Axes2
        % Delete the two photos just taken in the direction and reset the instruction.
        files = dir(fullfile(FolderPath, '*.jpg'));
        if length(files) >= 2
            LastTwoFiles = {files(end-1).name, files(end).name};
            LastTwoFilesPaths = fullfile(FolderPath, LastTwoFiles);
            for i = 1:2
                delete(LastTwoFilesPaths{i});
            end
            set(Inputbox, 'String', 'Please snap again');
        else
            set(Inputbox, 'String', 'Not enough files to undo');
        end
        snapCounter = snapCounter - 1;
        snapCounter = Switch(snapCounter);
    end

    function Reset(~, ~)                             % Reset the system
        % Deleting folder and images
        if ~isempty(FolderPath)                      
            files = dir(fullfile(FolderPath, '*.jpg'));
            for i = 1:length(files)
                filepath = fullfile(FolderPath, files(i).name);
                delete(filepath)
            end
            rmdir(FolderPath)
            FolderPath = '';
        end
        % Resetting bottoms and textboxs, and emptying the display of Axes2
        Inputbox = findobj(gcf, 'Tag', 'Inputbox');
        Cb = findobj(gcf, 'Tag', 'Cb');
        Output_instruction = findobj(gcf, 'Tag', 'Output_instruction');
        Output_complete = findobj(gcf, 'Tag', 'Output_complete');
        set(Inputbox, 'String', 'Enter Name Here'); 
        set(Cb, 'Value', 0);
        set(Output_instruction, 'String', 'Positive Face');
        set(Output_complete, 'String', ' ');
        cla(Axes2);                                 
    end

    function model = LoadedModel(model_fixed_name)    % Load the latest trained facial recognition neural network model.
        ModelPath = 'network';
        models = dir(fullfile(ModelPath, [model_fixed_name, '*.mat']));
        if isempty(models)
            defaultModelFile = fullfile(ModelPath, 'MobileNet_faceRecognition.mat');
            loadedModel = load(defaultModelFile);
            model = loadedModel.model;
        else
            [~, idx] = max([models.datenum]);
            loadedModelFile = models(idx);
            loadedModel = load(fullfile(ModelPath, loadedModelFile.name));
            model = loadedModel.model;
        end
    end

    function Submit(~, ~)                             % make the collected data to be trained by pre-trained neural network.
        if isempty(FolderPath)                        % only happen when folder is not empty.
            Inputbox = findobj(gcf, 'Tag', 'Inputbox');
            set(Inputbox, 'String', 'Please create a folder first.');
            return;
        end
        % load pre-trained neural network
        model = LoadedModel('New_MobileNet_faceRecognition_');   
        lgraph = layerGraph(model);

        % set training set and test set
        FaceDataPath = '/Users/yangsansui/Desktop/project/Dataset/FaceDataset';    
        imds = imageDatastore(FaceDataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        [imdsTrain, imdsTest] = splitEachLabel(imds, 8, 'randomized');

        % Data enhancement through data preprocessing
        pixelRange = [-5 5];    
        imageAugmenter = imageDataAugmenter( ...
            'RandXReflection',true, ...
            'RandXTranslation',pixelRange, ...
            'RandYTranslation',pixelRange, ...
            'RandRotation', [-15, 15]);
        augimdsTrain = augmentedImageDatastore([224 224],imdsTrain,'DataAugmentation',imageAugmenter,'ColorPreprocessing','gray2rgb');
        augimdsTest = augmentedImageDatastore([224 224],imdsTest,'DataAugmentation',imageAugmenter,'ColorPreprocessing','gray2rgb');

        % Adjust the structure of neural network， setting training parameters and train the network model
        original_numClasses = model.Layers(end-2).OutputSize;
        numClasses = original_numClasses + 1;
        newfc_Layer = fullyConnectedLayer(numClasses, 'Name', 'fc_func','WeightLearnRateFactor',1,'BiasLearnRateFactor',1,'WeightL2Factor',0.001);
        lgraph = replaceLayer(lgraph, 'fc_func', newfc_Layer);
        newClassLayer = classificationLayer('Name','new_classoutput');
        lgraph = replaceLayer(lgraph,'new_classoutput',newClassLayer);
        options = trainingOptions('adam', ...
            'ExecutionEnvironment', 'cpu', ...
            'InitialLearnRate', 1e-5, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.1, ...
            'LearnRateDropPeriod', 10, ...
            'MiniBatchSize', 8, ...
            'Shuffle','every-epoch', ...
            'Verbose', true, ...
            'MaxEpochs', 30, ...
            'Plots','training-progress');
        newModel = trainNetwork(augimdsTrain, lgraph, options);
        
        % save the model when accuracy is above 90%
        YPred = classify(newModel, augimdsTest);
        YValiation = imdsTest.Labels;
        accuracy = mean(YPred == YValiation);
        if accuracy > 0.9
            timeStamp = datetime("now");
            newModel_name = ['network/New_MobileNet_faceRecognition_', timeStamp, '.mat'];
            save(newModel_name, 'newModel');
            set(Output_complete, 'String', 'Completely!')
        else
            set(Output_complete, 'String', 'False.')
        end
    end


end

