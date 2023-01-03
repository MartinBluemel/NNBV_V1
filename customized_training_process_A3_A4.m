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


% numIterationsPerEpoch (N=numberOfData/miniBatchsize)
numObservations = numel(YTrain);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);

%total num of itarations during training
numiterations = numIterationsPerEpoch*numEpochs;

%set monitor update frequenzy 
validupdatefrequenzy = 64;

%numClasses
classes = categories(YTrain);


%Solver Modi
% 0 = Adam
% 1 = Sgdm
solver = 0;


%% A3
%array mit den zu untersuchenden Lernraten
varlearnrate = [0.1 0.01 0.001 0.0001 0.00001 0.000001];
%arrays für die Klassifizierungen je Solver
adanaccs = zeros(1,6);
sgdmaccs = zeros(1,6);

%Schleife fuer adamupdate mit varlearnrate
for i =1:6
adamaccs(i) = getaccuracy(miniBatchSize, ...
                                    numEpochs, ...
                                    varlearnrate(i), ...
                                    numIterationsPerEpoch, ...
                                    numiterations, ...
                                    classes, ...
                                    validupdatefrequenzy, ...
                                    XTrain,YTrain, ...
                                    XValid,YValid, ...
                                    imagestest,labelstest, ...
                                    dlnet, ...
                                    0);

end
disp(adamaccs);

%Schleife fuer sgdmupdate mit varlearnrate
for i =1:6
sgdmaccs(i) = getaccuracy(miniBatchSize, ...
                                    numEpochs, ...
                                    varlearnrate(i), ...
                                    numIterationsPerEpoch, ...
                                    numiterations, ...
                                    classes, ...
                                    validupdatefrequenzy, ...
                                    XTrain,YTrain, ...
                                    XValid,YValid, ...
                                    imagestest,labelstest, ...
                                    dlnet, ...
                                    1);

end
disp(sgdmaccs);

%plot für A3
semilogx(varlearnrate,adamaccs);                                  
hold on
semilogx(varlearnrate,sgdmaccs);                                  
grid on
title('μ=f(rate)')
xlabel('rate')
ylabel('Accuracy μ')
legend('adam','sgdm')
hold off

%% A4
learnRate = 0.001;                                                 
%various batchsizes
varbatchsize = [16 32 64 128 256];
%array für die benötigten Zeiten
elapsedtime = zeros(1,5);
%array für die Klassifizierungen
adamaccs = zeros(1,5);

%Schleife fuer adamupdate mit varbatchsize
for i = 1:5                                    
    % Zeitmessung starten 
    tStart = tic;                                                
    % neu berechnung der Anzahl von iterationen pro Epoche
    numIterationsPerEpoch = floor(numel(YTrain)/varbatchsize(i)); 
    numiterations = numEpochs * numIterationsPerEpoch;               
    adamaccs(i) = getaccuracy(varbatchsize(i), ...
                                    numEpochs, ...
                                    learnRate, ...
                                    numIterationsPerEpoch, ...
                                    numiterations, ...
                                    classes, ...
                                    validupdatefrequenzy, ...
                                    XTrain,YTrain, ...
                                    XValid,YValid, ...
                                    imagestest,labelstest, ...
                                    dlnet,0);
    % Zeitmessung beenden
    elapsedtime(i) = toc(tStart);                                
    tStart = 0;
    disp(elapsedtime(i));
end
%% Plot A4
disp(adamaccs)
tiledlayout(1,2)
nexttile
plot(varbatchsize,adamaccs);                                       
title('μ=f(batchsize)')
xlabel('batch size')
ylabel('Accuracy μ ')
grid on
hold on

nexttile
plot(varbatchsize,elapsedtime);                                   
title('t=f(batchsize)')
xlabel('batchsize')
ylabel('Time t in s ')
grid on
hold off






%% Train neural network
%ausgliedern des Trainings als Funktion
function accuracy = getaccuracy(miniBatchSize, ...
                                    numEpochs, ...
                                    learnRate, ...
                                    numIterationsPerEpoch, ...
                                    numiterations, ...
                                    classes, ...
                                    validupdatefrequenzy, ...
                                    XTrain,YTrain, ...
                                    XValid,YValid, ...
                                    imagestest,labelstest, ...
                                    dlnet, ...
                                    solver)

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
    
    
    %Trainingskurven haben 8-fache Aufloesung im Monitor
    trainupdatefrequenzy = validupdatefrequenzy/8;
    
    
    % initialize the average gradients and squared average gradients
    % averageGrad
    averageGrad = [];
    % averageSqGrad
    averageSqGrad = [];
    %Initialize the parameter velocities for the first iteration
    vel = [];
    %numclasses
    numClasses = numel(classes);
    
    %Monitor Status
    monitor.Status = "Runnig";
    
    % initailation itertationcounter
    iteration = 0;
    
    %Zaehler fuer Monitorupdate
    t = 0;
    v = 0;
    
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
    
            %Auswahl des Solvers
            if solver == 0
                % Update the network parameters using the optimizer Adam
                [dlnet,averageGrad,averageSqGrad] = adamupdate(dlnet,grad,averageGrad,averageSqGrad,iteration,learnRate);
            end
            if solver == 1
                % Update the network parameters using the optimizer Adam
                [dlnet,vel] = sgdmupdate(dlnet,grad,vel,learnRate);
            end
            % Calculate accuracy & show the training progress. 
            if t > trainupdatefrequenzy || t == 0
                t = 0;
               
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
end

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