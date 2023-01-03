%% Load Training Data & define class catalog & define input image size
disp('Loading training data...')
% download from MNIST-home page or import dataset from MATLAB
% https://www.mathworks.com/help/deeplearning/ug/data-sets-for-deep-learning.html
% http://yann.lecun.com/exdb/mnist/

% Specify training and validation data
% Recommended naming >>>
% Train: dataset for training a neural network
% Test: dataset for test a trained neural network after training process
% Valid: dataset for test a trained neural network during training process
% X: input / for Classification: image
% Y: output / for Classification: label
% for example: XTrain, YTrain, XTest, YTest, XValid, YValid

oldpath = addpath(fullfile(matlabroot,'examples','nnet','main'));
filenameImagesTrain = 'train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'train-labels-idx1-ubyte.gz';
filenameImagesTest = 't10k-images-idx3-ubyte.gz';
filenameLabelsTest = 't10k-labels-idx1-ubyte.gz';

imagestrain  = processImagesMNIST(filenameImagesTrain);
labelstrain  = processLabelsMNIST(filenameLabelsTrain);
imagestest   = processImagesMNIST(filenameImagesTest);
labelstest   = processLabelsMNIST(filenameLabelsTest);
%nur zum Dimensionsverstaendnis
whos imagestrain
whos labelstrain

path(oldpath);

XImages = imagestrain;
YLabels = labelstrain;


%Aufteilung der gegebenen Trainingsdaten in Validierungs und Trainingsdaten
[trainInd,validInd,testInd] = dividerand(numel(YLabels),0.8,0.2,0);         
XTrain = XImages(:,:,:,trainInd);
YTrain = YLabels(trainInd);                                                                            
XValid = XImages(:,:,:,validInd);
YValid = YLabels(validInd);
%testInd ist hier auf 0 gesetzt da wir ja schon einen Testdatensatz gegeben
%haben, alle gegebenen Daten sind jetzt ca. wie folgt aufgeteilt: 70%
%Training, 15% Validierung, 15% Test

%nur zum Aufteilungs- und Dimensionsverstaendnis 
whos XTrain
whos YTrain
whos XValid
whos YValid


%Network arcitecure
inputSize = [28 28 1];
layer0 = imageInputLayer(inputSize);

outputSize = 256;
layer1 = fullyConnectedLayer(outputSize);

layer2 = reluLayer;

layer3 = fullyConnectedLayer(10);

layer4 = softmaxLayer;

layer5 = classificationLayer;


NN_layers = [
    layer0
    layer1
    layer2
    layer3
    layer4
    layer5
];

% visualize the neural network
analyzeNetwork(NN_layers)
%% Specify Training Options (define hyperparameters)

% miniBatchSize
% numEpochs
% learnRate 
% executionEnvironment
% numIterationsPerEpoch 
% solver "sgdm" "rmsprop" "adam"

%Options
options = trainingOptions("adam");
    options.MaxEpochs = 10;
    options.ValidationData = {XValid,YValid};
    options.Plots = 'training-progress';


% training on CPU or GPU(if available);
% 'auto': Use a GPU if one is available. Otherwise, use the CPU.
% 'cpu' : Use the CPU
% 'gpu' : Use the GPU.
% 'multi-gpu' :Use multiple GPUs
% 'parallel :

%%  Train neural network
% training
trainednet = trainNetwork(XTrain,YTrain,NN_layers,options);

%% test neural network & visualization 
% Calculate accuracy
ytest = classify(trainednet,imagestest);

accuracy = mean(ytest == labelstest);
disp('Calculated accuracy ='),disp(accuracy);


