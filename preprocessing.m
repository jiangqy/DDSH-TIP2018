function preprocessing(dataname)
switch dataname
    case 'CIFAR-10'
        if ~exist('./data/CIFAR-10/CIFAR-10.mat', 'file')
            if ~exist('./data/CIFAR-10/cifar-10-batches-mat', 'file')
                % not download yet
                if ~exist('./data/CIFAR-10/cifar-10-matlab.tar.gz', 'file')
                    download_cifar10();
                end
                gunzip('./data/CIFAR-10/cifar-10-matlab.tar.gz');
                untar('./data/CIFAR-10/cifar-10-matlab.tar','./data/CIFAR-10/');
                delete('./data/CIFAR-10/cifar-10-matlab.tar');
            end
            sampling_cifar10();
        end
end
end

function download_cifar10()
try
    urlwrite('https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz', ...
        './data/CIFAR-10/cifar-10-matlab.tar.gz');
catch 
    disp('error in downloading, please download agian or download manually');
end
end

function sampling_cifar10()

XAll = [];
LAll = [];
for i = 1: 5
    clear data labels batch_label;
    load(['./data/CIFAR-10/cifar-10-batches-mat/data_batch_' num2str(i) '.mat']);
    data = reshape(data', [32, 32, 3, 10000]);
    data = permute(data, [2, 1, 3, 4]);
    XAll = cat(4, XAll, data);
    LAll = cat(1, LAll, labels);
end

clear data labels;
load('./data/CIFAR-10/cifar-10-batches-mat/test_batch.mat');

data = reshape(data', [32, 32, 3, 10000]);
data = permute(data, [2, 1, 3, 4]);

XAll = cat(4, XAll, data);
LAll = cat(1, LAll, labels);


XTrain = [];
XTest = [];
XDatabase = [];
trainL = [];
testL = [];
databaseL = [];
for label = 0: 9
    index = find(LAll == label);
    N = size(index, 1);
    perm = randperm(N);
    index = index(perm);
    
    data = XAll(:, :, :, index(1: 100));
    labels = LAll(index(1: 100));
    XTest = cat(4, XTest, data);
    testL = cat(1, testL, labels);
    
    index(1: 100) = [];
    
    data = XAll(:, :, :, index(1: 500));
    labels = LAll(index(1: 500));
    
    XTrain = cat(4, XTrain, data);
    trainL = cat(1, trainL, labels);
    
    data = XAll(:, :, :, index(1: end));
    labels = LAll(index(1: end));
    XDatabase = cat(4, XDatabase, data);
    databaseL = cat(1, databaseL, labels);
end

num_train = size(trainL, 1);
perm = randperm(num_train);
XTrain = XTrain(:, :, :, perm);
trainL = trainL(perm);

save('data/CIFAR-10/CIFAR-10.mat', 'XTrain', 'trainL', 'XDatabase', ...
    'databaseL', 'XTest', 'testL');

end