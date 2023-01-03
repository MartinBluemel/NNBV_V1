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


%Aufteilung der gegebenen Traeningsdaten in Validierungs und Traeningsdaten
[trainInd,validInd,testInd] = dividerand(numel(YLabels),0.8,0.2,0);         
XTrain = XImages(:,:,:,trainInd);
YTrain = YLabels(trainInd);                                                                            
XValid = XImages(:,:,:,validInd);
YValid = YLabels(validInd);
%testInd ist hier auf 0 gesetzt da wir ja schon einen Testdatensatz gegeben
%haben, alle gegebenen Daten sind jetzt ca. wie folgt aufgeteilt: 70%
%Traening, 15% Validierung, 15% Test

%numClasses
classes = categories(YTrain);
numClasses = numel(classes);

%% define network (dlnet)
inputSize = [28 28 1];
layer0 = imageInputLayer(inputSize,"Normalization","none");

outputSize = 256;
layer1 = fullyConnectedLayer(outputSize);

layer2 = reluLayer;

layer3 = fullyConnectedLayer(10);

layer4 = softmaxLayer;


NN_layers = [
    layer0
    layer1
    layer2
    layer3
    layer4
];

% convert to a layer graph
lgraph = layerGraph(NN_layers);
% Create a dlnetwork object from the layer graph.
dlnet = dlnetwork(lgraph);
% visualize the neural network
analyzeNetwork(dlnet)




%% Specify Training Options (define hyperparameters)
% miniBatchSize
miniBatchSize = 128;

% numEpochs
numEpochs = 10;

% learnRate 
learnRate = 0.001;

% numIterationsPerEpoch (N=numberOfData/miniBatchsize)
numObservations = numel(YTrain);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);

%total num of itarations during training
numiterations = numIterationsPerEpoch*numEpochs;

%Trainings Monitor
%from "https://de.mathworks.com/help/deeplearning/ug/monitor-custom-training-loop-progress.html"
monitor = trainingProgressMonitor;
monitor.Info = ["LearningRate","Epoch","Iteration","TrainingAccuracy","ValidationAccuracy"];
monitor.Metrics = ["TrainingLoss","ValidationLoss","TrainingAccuracy","ValidationAccuracy"];

monitor.XLabel = "Iteration";
monitor.Status = "Configuring";
%monitor.Progress = 0;

groupSubPlot(monitor,"Loss",["TrainingLoss","ValidationLoss"]);
groupSubPlot(monitor,"Accuracy(%)",["TrainingAccuracy","ValidationAccuracy"]);

%set monitor update frequenzy 
%Trainingskurven haben 8-fache Aufloesung im Monitor
validupdatefrequenzy = 64;
trainupdatefrequenzy = validupdatefrequenzy/8;
%Zähler für die Monitorfrequenz
t = 0;
v = 0;


%% Train neural network
% initialize the average gradients and squared average gradients
% averageGrad
averageGrad = [];
% averageSqGrad
averageSqGrad = [];

%Monitor Status
monitor.Status = "Runnig";

% initailation itertationcounter
iteration = 0;

%from "https://de.mathworks.com/help/deeplearning/ref/adamupdate.html"
% "for-loop " for training
for epoch = 1:numEpochs 

    % Shuffle trainings data
    idx = randperm(numel(YTrain));
    XTrain = XTrain(:,:,:,idx);
    YTrain = YTrain(idx); 
    % Shuffle validation data
    %idxv = randperm(numel(YValid));
    %XValid = XValid(:,:,:,idxv);
    %YValid = YVAlid(idxv);


    % update learnable parameters based on mini-batch of data
    for i = 1:numIterationsPerEpoch

        %iterationcounter
        iteration = iteration + 1;

        % Read mini-batch of data and convert the labels to dummy variables.
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        X = XTrain(:,:,:,idx);
        
        Y = zeros(numClasses, miniBatchSize,"single");                                  
            for c = 1:numClasses
                 Y(c,YTrain(idx)==classes(c)) = 1;
            end


        % Convert mini-batch of data to a dlarray.
        dlX = dlarray(single(X),"SSCB");
   

        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients helper function.
        [grad,loss,dlYPred] = dlfeval(@modelGradients,dlnet,dlX,Y);

        % Update the network parameters using the optimizer Adam
        [dlnet,averageGrad,averageSqGrad] = adamupdate(dlnet,grad,averageGrad,averageSqGrad,iteration);
        
        % Calculate accuracy & show the training progress. 
        if t > trainupdatefrequenzy || t == 0
            t = 0;
            %damit keine negative Klassifizierungsgenauigkeit im Monitor
            %aufgezeichnet wird
            taccuracy = 0;
            if loss <= 1
                taccuracy = 1-loss;
            end
            % Record training loss and accuracy.
            recordMetrics(monitor,iteration, ...
                TrainingLoss=loss, ...
                TrainingAccuracy=taccuracy*100)
            
            % Update learning rate, epoch, and iteration information values.
            updateInfo(monitor, ...
                LearningRate=learnRate, ...
                Epoch=string(epoch) + " of " + string(numEpochs), ...
                Iteration=string(iteration) + " of " + string(numiterations), ...
                TrainingAccuracy=string((1-loss)*100) + "%");
        end
        t = t + 1;

        % option: validation
        if v > validupdatefrequenzy || v == 0
            v = 0;  
            
            yvali = predict(dlnet,dlarray(single(XValid),'SSCB'));    
            [~,idx] = max(extractdata(yvali),[],1);
            yvali = classes(idx);
    
            valiaccuracy = mean(yvali == YValid); 
         
       
            % Record validation loss and accuracy.
            recordMetrics(monitor,iteration, ...
                               ValidationLoss=(1-valiaccuracy), ...
                               ValidationAccuracy=valiaccuracy*100);%*100 fuer prozent
         
        end
        v = v + 1;
        monitor.Progress = 100*iteration/numiterations;

    end
end

%Monitor Status
monitor.Status = "Training complete";



%% test neural network & visualization 
%from "https://de.mathworks.com/help/deeplearning/ref/adamupdate.html"
yvali = predict(dlnet,dlarray(single(imagestest),'SSCB'));    
[~,idx] = max(extractdata(yvali),[],1);
yvali = classes(idx);

accuracy = mean(yvali == labelstest);
disp('Calculated accuracy during test ='),disp(accuracy);

% x = double(labelstest(1))-1;
% disp(x)
% y = str2num(yvali{1});
% disp(y)

%array mit den Klassifikationen je Ziffer
accs = zeros(1,10);
%for schleife zum sortieren der Label und Predictions nach den Ziffern
for j =1:10

    z = 1;
    label1 = zeros(1,z);
    vali1 = zeros(1,z);
    for i = 1:numel(labelstest)
        
        if  double(labelstest(i))-1 == j-1
            label1(1,z) = double(labelstest(i))-1;
            vali1(1,z) = str2num(yvali{i});
            z = z+1;
        end
    end
    %disp(label1)
    disp(length(label1));
    %disp(vali1)
    currentacc = mean(label1 == vali1);
    %disp(currentacc);
    accs(1,j) = currentacc;

end
disp(accs)
%plot Aufgabe 2
hold on
grid on
scatter(1:8,accuracy,10000,"red","_");
scatter([0 1 2 3 4 5 6 7 8 9],accs,"blue");
scatter(1:1,max(accs),1000,"green",".")

title('μ=f(Ziffer)')
xlabel('Ziffer')
ylabel('Accuracy μ')

hold off

%% Define Model Gradients Function
% 
function [gradients,loss,dlYPred] = modelGradients(dlnet,dlX,Y)

    % forward propagation 
    dlYPred = forward(dlnet,dlX);
    % calculate loss -- varies based on different requirement
    loss = crossentropy(dlYPred,Y);
    % calculate gradients 
    gradients = dlgradient(loss,dlnet.Learnables);
    
end