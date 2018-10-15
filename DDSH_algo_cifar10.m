function DDSH_algo_cifar10()
%% set seeds
seed = 0;
rng('default');
rng(seed);

addpath(genpath('utils'));
gpuDevice(7);
run('matconvnet/setup');

%% set parameters
% basic parameters
bits = [12, 24, 32, 48];
dataname = 'CIFAR-10';
logfile = ['log/DDSH-' dataname '-seed-' num2str(seed) '-' datestr(now)];
mkdir(logfile);

%% download datasets
preprocessing(dataname);

% load data
[dataset, param] = load_data(dataname);
param.bits = bits;
param.dataname = dataname;
param.logfile = logfile;
% description
param.desc = 'DDSH method on CIFAR-10 dataset.';

%% DDSH procedure
for bit = bits
    param.bit = bit;    
    % training procedure
    [net, loss] = trainDDSH(dataset.XTrain, dataset.trainL, param);
    % encoding procedure
    tB = encoding(dataset.XTest, net, bit);
    dB = encoding(dataset.XDatabase, net, bit);
    
    save([logfile '/hashCodes-' num2str(bit) '.mat'], 'tB', 'dB');
    % compressing binary codes
    tB = compactbit(tB > 0);
    dB = compactbit(dB > 0);
    % evaluation
    testL = dataset.testL;
    databaseL = dataset.databaseL;
    hr = callHRLabel(testL, databaseL, tB, dB, param);
    
    result.map = hr.map;
    result.loss = loss;
    save([logfile '/result-' num2str(bit) '.mat'], 'result');
    disp(['#bit: ' num2str(bit) ', map: ' num2str(result.map, '%.4f')]);    
end

end

function B = encoding(X, net, bit)
num_data = size(X, 4);
batch_size = 256;

B = zeros(num_data, bit);
for j = 0:ceil(num_data/batch_size)-1
    ix = (1+j*batch_size):min((j+1)*batch_size, num_data);
    im = single(X(:,:,:,ix));
    im_ = imresize(im,net.normalization.imageSize(1:2));
    im_ = im_ - repmat(net.normalization.averageImage,1,1,1,size(im_,4));
    im_ = gpuArray(im_);
    res = vl_simplenn(net, im_) ;
    features = squeeze(gather(res(end).x))' ;
    B(ix,:) = features ;
end

B = sign(B);

end
