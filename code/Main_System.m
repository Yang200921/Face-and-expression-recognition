function Main_System    % For sharing variables between parent and nested functions
%%
% To run this program on your own computer, the areas that need to be modified include:
% Ensure that all required MATLAB toolboxes, including the deepNetworkDesigner and Computer Vision Toolbox, are installed, along with the MobileNetV2 neural network model.
% Line 58, videoinput() should adapt to your device.
% Line 269 and 353, faceDatasetPath should be set to the dataset path in your computer.
%%
%=============================================================================%
%===================             Honor Project           =====================%
%===================               ME41002               =====================%
%=====        Camera-based Face and Expression Recognition system        =====%
%=====                        Authors: Ziqi Yang                         =====%
%=====                       Current version: 6.0                        =====%
%=====                         Date: April 2024                          =====%
%=============================================================================%

%-----------------------------------------------------------------------------%
%--------------------           Version Histroy          ---------------------%
%-----------------------------------------------------------------------------%
%                                                                             %
% REVISIONS FOR VERSION 1.0                                                   %
% Builded a Gui and did face detection, showing with a yellow rectangle.      %
% Added trackers to track a specially choosen face.                           %
%                                                                             %
%                                                                             %
% REVISIONS FOR VERSION 2.0:                                                  %
% Temporarily deleted the tracker for tracking a specially choosen face.      %
% Improve the Gui in showing more images and adding functions.                %
% Did an automatic screenshot for cropping the face.                          %
% Cropped the face for enhancing image and do face recognization.             %
% After turning off the face detection, clear every axes except Live.         %
%                                                                             %
%                                                                             %
% REVISIONS FOR VERSION 3.0                                                   %
% Builded a Gui and did face detection, showing with a yellow rectangle.      %
% Added trackers to track a specially choosen face.                           %
%                                                                             %
% REVISIONS FOR VERSION 4.0                                                   %
% Restrict the bbox within frame.                                             %
% Multiple faces recognition and cropped.                                     %
%                                                                             %
% REVISIONS FOR VERSION 5.0                                                   %
% Adjusted the Gui design.                                                    %
% Designed and trained a Mobilenet V2 model to do face recognition.           %
%                                                                             %
% REVISIONS FOR VERSION 6.0                                                   %
% Adjusted the Gui design.                                                    %
% Designed and trained a Mobilenet V2 model to do expression detection.       %
%-----------------------------------------------------------------------------%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Open Camera %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Build a Gui %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fig = figure('Position',[100,150,980,500]);

Pnl1 = uipanel(Fig,'Position',[0.05,0.05,0.35,0.9]);   
Pnl2 = uipanel(Fig,'Position',[0.425,0.05,0.15,0.9]);   
Pnl3 = uipanel(Fig,'Position',[0.60,0.05,0.35,0.9]);   

Axes1 = axes(Pnl1,'Position',[0,0.6,1,0.3],'Visible', 'off');
Axes2 = axes(Pnl1,'Position',[0,0.1,0.45,0.3], 'Visible', 'off');
Axes3 = axes(Pnl1,'Position',[0.55,0.1,0.45,0.3],'Visible', 'off');
Axes4 = axes(Pnl3,'Position',[0,0.6,0.45,0.3],'Visible', 'off');
Axes5 = axes(Pnl3,'Position',[0.55,0.6,0.45,0.3], 'Visible', 'off');
Axes6 = axes(Pnl3,'Position',[0,0.1,0.45,0.3],'Visible', 'off');
Axes7 = axes(Pnl3,'Position',[0.55,0.1,0.45,0.3],'Visible', 'off');
axis off;

Bt1 = uicontrol(Pnl2,'style','togglebutton','String','Face Detection','Fontsize',10,'Units','normalized','Position',[0.1,0.85,0.8,0.05],'Callback',@FaceDetection);
Bt2 = uicontrol(Pnl2,'style','togglebutton','String','Posture Detection','Fontsize',10,'Units','normalized','Position',[0.1,0.75,0.8,0.05],'Callback',@PostureDetection);
Bt3 = uicontrol(Pnl2,'style','togglebutton','String','Crop','Fontsize',10,'Units','normalized','Position',[0.1,0.65,0.8,0.05],'Callback',@Crop);
Bt4 = uicontrol(Pnl2,'style','togglebutton','String','Multi-function','Fontsize',10,'Units','normalized','Position',[0.1,0.55,0.8,0.05],'Callback',@Multi_function);
Bt5 = uicontrol(Pnl2,'style','togglebutton','String','Expression Detection','Fontsize',10,'Units','normalized','Position',[0.1,0.45,0.8,0.05],'Callback',@ExpressionDetection);
Bt6 = uicontrol(Pnl2,'style','togglebutton','String','Face Recognization','Fontsize',10,'Units','normalized','Position',[0.1,0.35,0.8,0.05],'Callback',@FaceRecognization);
Bt7 = uicontrol(Pnl2,'style','togglebutton','String','Auto shot','Fontsize',10,'Units','normalized','Position',[0.1,0.25,0.8,0.05],'Callback',@Autoshot);

lb1 = uicontrol(Pnl1,'style','text','String','Live','Fontsize',14,'Units','normalized','Position',[0.35,0.925,0.3,0.05]);
lb2 = uicontrol(Pnl1,'style','text','String','Face Detection','Fontsize',12,'Units','normalized','Position',[0.075,0.425,0.3,0.05]);
lb3 = uicontrol(Pnl1,'style','text','String','Posture Detection','Fontsize',12,'Units','normalized','Position',[0.575,0.425,0.4,0.05]);
lb4 = uicontrol(Pnl3,'style','text','String','Cropped Face','Fontsize',12,'Units','normalized','Position',[0.075,0.925,0.3,0.05]);
lb5 = uicontrol(Pnl3,'style','text','String','Expression Detection','Fontsize',12,'Units','normalized','Position',[0.575,0.925,0.4,0.05]);
lb6 = uicontrol(Pnl3,'style','text','String','Face Recognition','Fontsize',12,'Units','normalized','Position',[0.075,0.425,0.4,0.05]);
lb7 = uicontrol(Pnl3,'style','text','String','test','Fontsize',12,'Units','normalized','Position',[0.075,0.0125,0.3,0.05]);
lb8 = uicontrol(Pnl3,'style','text','String','match','Fontsize',12,'Units','normalized','Position',[0.625,0.0125,0.3,0.05]);

ef1 = uicontrol(Pnl3,'style','edit','String','Expression','Fontsize',12,'Units','normalized','Position',[0.625,0.5125,0.3,0.05],'Tag','Expression');
ef2 = uicontrol(Pnl3,'style','edit','String','Who','Fontsize',12,'Units','normalized','Position',[0.625,0.425,0.3,0.05],'Tag','Who');

drawnow                                              % Draw the GUI with set components

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% face detection parameters
flag_detection = 1;
flag_Bt1 = 0;
faceDetector = vision.CascadeObjectDetector();       % Create a cascade object detector for detecting positive face
gap = 2;                                             % Speed up the calculation
prev_num_faces = 0;
Shot = [];


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while ishandle(camera)
      frame = getsnapshot(vid);
      frame = im2double(frame);
      imshow(frame,'Parent',Axes1)                   % Display the frame in Live Axes

      % Live & Face Detection
      if flag_Bt1                                    % If FaceDetection bottom is pressed, start Face Detection.
          
          % Execute Autoshot function when the number of recognized faces in Live is changing.
          bbox = facedetect(frame);                  % Detect faces in the frame by customized function
          num_faces = size(bbox, 1);
          if num_faces ~= prev_num_faces
              flag_detection = 1;
          end
          if ~isempty(bbox) && flag_detection 
              Shot = frame;
              autoShot = Autoshot(Shot, bbox);       % Call the Autoshot function to insert rectangles on screenshot to frame faces
              imshow(autoShot, 'Parent', Axes2);
              flag_detection = 0;
              set(Bt7, 'Value', 1);
              prev_num_faces = num_faces;
          end
          hold(Axes1,'on')

          % Rectangle the recognized faces in live Axes
          for i = 1:num_faces
              rc = bbox(i, :) + [-bbox(i, 3)/4, -bbox(i, 4)/4, bbox(i, 3)/2, bbox(i, 4)/2];
              rectangle('Position', rc, 'Curvature', 0, 'LineWidth', 2, 'LineStyle', '--', 'EdgeColor', 'y', 'Parent', Axes1);
          end
          hold(Axes1,'off')


      % Reset the system when the face detection button is unpressed.
      else    
          cla(Axes2);
          cla(Axes3);
          cla(Axes4);
          cla(Axes5);
          cla(Axes6);
          cla(Axes7);
          set(Bt2, 'Value', 0);
          set(Bt3, 'Value', 0);
          set(Bt4, 'Value', 0);
          set(Bt5, 'Value', 0);
          set(Bt6, 'Value', 0);
          set(Bt7, 'Value', 0);
          Who = findobj(gcf, 'Tag', 'Who');
          Expression = findobj(gcf, 'Tag', 'Expression');
          set(Expression, 'String', 'Expression');
          set(Who, 'String', 'Who');
          prev_num_faces = 0;
          Faces = {};
      end
      drawnow
end


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Nested Functions %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% face detection
    function FaceDetection(~,~)                      % Determine whether to perform face detection, based on the button state.
        flag_Bt1 = get(Bt1,'Value');
        flag_detection = 1;
    end

    function bbox = facedetect(frame)                % Detect the faces and record the positions
        frame = frame(1:gap:end,1:gap:end,:);
        bbox = step(faceDetector,frame);
        bbox = bbox*gap;
    end

    function autoShot = Autoshot(Shot, bbox)         % Insert square boxes in the screenshot to show the face positions.
        for i = 1:size(bbox, 1)
            bbox_autoshot = bbox(i, :) + [-bbox(i, 3)/4, -bbox(i, 4)/4, bbox(i, 3)/2, bbox(i, 4)/2];
            if bbox_autoshot(1) > 1 && bbox_autoshot(2) > 1 && bbox_autoshot(3) < size(Shot, 2) - 1 && bbox_autoshot(4) < size(Shot, 1) - 1
                autoShot = insertShape(Shot, 'Rectangle', bbox(i, :), 'LineWidth', 5);
            else
                return
            end
        end
    end

% Crop face
    function Crop(~,~)                               % Crop the face images according to the face coordinates obtained by face detection and save in Faces.
        Faces = cell(1, size(bbox, 1)); 
        localShot = Shot;
        for i = 1:size(bbox, 1)
            bbox_crop = bbox(i, :) + [-bbox(i, 3)/4, -bbox(i, 4)/4, bbox(i, 3)/2, bbox(i, 4)/2];
            Face = imcrop(localShot, bbox_crop);
            Faces{i} = Face;
        end
        for i = 1:numel(Faces)                      % Rollingly show the face images in Axes4.
            imshow(Faces{i},'Parent',Axes4)
            pause(2)
        end
    end

% Face Recognization
    function model = LoadedModel(model_fixed_name)   % Load the latest trained model.
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

    % The probability of the predicted label exceeding the threshold is a prediction label; otherwise, set the predicted label as none.
    function YPred = predThresh(predLabels, Pred_probability, threshold)  
        YPred = cell(size(Pred_probability, 1), 1);
        for i = 1:size(Pred_probability, 1)
            if Pred_probability(i,1) >= threshold
                YPred{i} = predLabels{i};
            else
                YPred{i} = 'none';
            end
        end
    end

    function saveFaceImages(Faces, tempDir, flag)    % Create a temporary folder to store face images for neural network use via imageDatastore.
        if ~exist(tempDir, 'dir')
            mkdir(tempDir);
        end
        for i = 1:numel(Faces)
            if flag
                Faces{i} = rgb2gray(Faces{i});
            end
            tempfileName = sprintf('%d.jpg', i);
            tempfilePath = fullfile(tempDir, tempfileName);
            imwrite(Faces{i}, tempfilePath);
        end
    end

    function FaceRecognization(~, ~)                 % Recognize faces and display results
        % load dataset with labels
        FaceDataPath = '/Users/yangsansui/Desktop/project/Dataset/FaceDataset';    
        imds_faces = imageDatastore(FaceDataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

        % Initialize directories and load face images into MATLAB
        tempDir = fullfile(FaceDataPath,'temp_Faces');
        saveFaceImages(Faces, tempDir,1)
        imds = imageDatastore(tempDir);

        % Load pre-trained model and prepare data for prediction
        model = LoadedModel('New_MobileNet_faceRecognition_');
        augimds = augmentedImageDatastore([224 224],imds,'ColorPreprocessing','gray2rgb');

        % Perform predictions using the model
        labels = categories(imds_faces.Labels);
        PPred = predict(model, augimds);
        [Pred_probability, maxIdx] = max(PPred, [], 2); 
        predictedLabels = labels(maxIdx);

        % Apply a threshold and display results
        threshold = 0.8;
        YPred = predThresh(predictedLabels, Pred_probability, threshold);
        Who = findobj(gcf, 'Tag', 'Who');

        % Display faces and prediction results in the GUI
        for i = 1:numel(Faces)
            cla(Axes6);
            cla(Axes7);
            imshow(Faces{i},'Parent',Axes6)
            currentYPred = YPred(i);
            YPred_str = char(currentYPred);
            facefolderPath = fullfile(FaceDataPath,YPred_str);
            Files = dir(fullfile(facefolderPath, '*.pgm'));
            if isempty(Files)
                Files = dir(fullfile(facefolderPath, '*.jpg'));
            end
            if ~isempty(Files)
                img = imread(fullfile(facefolderPath, Files(1).name));
                imshow(img,'Parent',Axes7)
                set(Who, 'String', YPred_str);
            else
                set(Who, 'String', 'None');
            end
            pause(2)
        end 
        rmdir(tempDir, 's');
    end

% Expression Detection
    function ExpressionDetection(~, ~)               % Recognize expressions and display results
        % Set up temporary directory for face images
        Expression = findobj(gcf, 'Tag', 'Expression');
        tempDir = 'temp_Faces';
        saveFaceImages(Faces, tempDir,0)
        imds = imageDatastore(tempDir);

        % Load the pre-trained expression recognition model and prepare image data for classification
        loadedmodel = load('network/MobileNet_expressionRecognition.mat');
        model = loadedmodel.model;
        augimds = augmentedImageDatastore([224 224],imds,'ColorPreprocessing','gray2rgb');

        % Classify expressions using the loaded model and display each face and its predicted expression in the GUI
        YPred = classify(model, augimds);
        disp(YPred);
        for i = 1:numel(Faces)
            cla(Axes5);
            currentYPred = YPred(i);
            YPred_str = char(currentYPred);
            set(Expression, 'String', YPred_str);
            imshow(Faces{i},'Parent',Axes5)
            pause(2)
        end 
        rmdir(tempDir, 's');
    end

% Multi-function
    function Multi_function(~, ~)                    % Crop face, recognize face and expressions, and display results in one function without pressing other buttom 
        % Save face images in temporary directory and rollingly display them in cropped face Axes
        Crop()
        tempDir = 'temp_Faces';
        saveFaceImages(Faces, tempDir,1)

         % Load and augment images for recognition
        imds = imageDatastore(tempDir);
        augimds = augmentedImageDatastore([224 224],imds,'ColorPreprocessing','gray2rgb');
        FaceDataPath = fullfile('/Users/yangsansui/Desktop/project/Dataset/FaceDataset');    
        imds_faces = imageDatastore(FaceDataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

        % Face recognition
        model_fr = LoadedModel('New_MobileNet_faceRecognition_');
        labels = categories(imds_faces.Labels);
        PPred = predict(model_fr, augimds);
        [Pred_probability, maxIdx] = max(PPred, [], 2); 
        predictedLabels = labels(maxIdx);
        threshold = 0.95;
        YPred = predThresh(predictedLabels, Pred_probability, threshold);
        Who = findobj(gcf, 'Tag', 'Who');

        % display the face recognition results in corresponding box and Axes
        for i = 1:numel(Faces)
            cla(Axes6);
            cla(Axes7);
            imshow(Faces{i},'Parent',Axes6)
            currentYPred = YPred(i);
            YPred_str = char(currentYPred);
            facefolderPath = fullfile(FaceDataPath,YPred_str);
            Files = dir(fullfile(facefolderPath, '*.pgm'));
            if isempty(Files)
                Files = dir(fullfile(facefolderPath, '*.jpg'));
            end
            if ~isempty(Files)
                img = imread(fullfile(facefolderPath, Files(1).name));
                imshow(img,'Parent',Axes7)
                set(Who, 'String', YPred_str);
            else
                set(Who, 'String', 'None');
            end
            pause(2)
        end 

        % Expression detection
        loadedmodel = load('network/MobileNet_expressionRecognition.mat');
        model_er = loadedmodel.model;
        YPred_er = classify(model_er, augimds);
        disp(YPred_er);
        Expression = findobj(gcf, 'Tag', 'Expression');

        % display the expression recognition results in corresponding box and Axes
        for i = 1:numel(Faces)
            cla(Axes5);
            imshow(Faces{i},'Parent',Axes5)
            currentYPred_er = YPred_er(i);
            YPred_str_er = char(currentYPred_er);
            set(Expression, 'String', YPred_str_er);
            pause(2)
        end 
        rmdir(tempDir, 's');
    end


end
%=============================================================================%
%==================                  END                    ==================%
%=============================================================================%