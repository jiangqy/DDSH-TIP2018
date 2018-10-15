function [net, loss] = trainDDSH(ITrain, trainL, param)
net = load('./data/imagenet-vgg-f.mat');

epochs = param.epochs;
iters = param.iters;
lr = param.lr;
batch_size = param.batch_size;
bit = param.bit;
numTrain = size(ITrain, 4);

sample_column = param.sample_column;
num_gamma = numTrain-sample_column;

net = net_structure(net, bit);

B = randn(numTrain,bit) > 0;
B = 2*B - 1;

loss = zeros(2 * epochs * iters, 1);
nB = B;
net = net_structure(net,bit);

count = 1;
% training procedure
for epoch = 1: epochs
    Omega = randsample(numTrain, sample_column);
    Gamma = setdiff([1: numTrain], Omega);
    
    sampleL = trainL(Omega, :);
    SS = trainL * sampleL' > 0;
    r = sum(SS(:)) / sum(1 - SS(:));    
    s1 = 1;    
    s0 = s1 * r;    
    SS = SS * (s1 + s0) - s0;
    
    % for nus-wide dataset, please set the negative similarity(SS) to 0 for
    % pairs which is constructed by the multi-label data samples.
    
    S_gamma = SS(Gamma, :);
    S_omega = SS(Omega, :);

    for t = 1:iters
        omega_time = tic;

        for k = 1:bit
            if k==1			
                lq=-2*bit*S_omega;				
            else
                lq=lq+2*nB(Omega,k-1)*nB(Omega,k-1)';	
            end   
            QQ=lq;
            QQ(1:sample_column+1:end)=0;
            Q=QQ;

            if k==1
                lp=bit*S_gamma;
            else
                lp=lp-B(Gamma,k-1)*nB(Omega,k-1)';
            end

            pp=-2*lp'*B(Gamma,k);
            p=pp;            

            % construct problem (5)		
            for i = 1:sample_column
                p(i)=2*p(i);
                for l=1:sample_column
                    p(i)=p(i)-2*(Q(i,l)+Q(l,i));
                end
            end
            Q = Q*4;
            % construct problem (6)			
            Q = [Q,p/2;p'/2,0];					
            % construct problem (8)            

            Q = Q/max(max(abs(Q)));            
            lambda = 10000;   			%sufficient large			

            P = chol(lambda*eye(sample_column+1)-Q);            

            %Update nB_Omega            
            nB(Omega,k)=Solve(P)*2-1;                           
        end
        
        omega_time = toc(omega_time);
        gamma_time = tic;
        nB_Omega = nB(Omega,:);
        nB_Gamma = nB(Gamma,:);
        index = randperm(num_gamma);
        ITrainGamma = ITrain(:,:,:,Gamma);        
        
        for k = 0:ceil(num_gamma/batch_size) - 1
            ix = index((1+k*batch_size):min((k+1)*batch_size,num_gamma));
            im = single(ITrainGamma(:,:,:,ix));
                
            im_ = imresize(im,net.normalization.imageSize(1:2));
            im_ = im_ - repmat(net.normalization.averageImage,1,1,1,size(im_,4));
            im_ = gpuArray(im_);
            
            res = vl_simplenn(net,im_);
            U0 = squeeze(gather(res(end).x))' ;
            nB_Gamma(ix,:) = tanh(U0);
            dJdB_g = -2*bit*S_gamma*nB_Omega + 2*nB_Gamma* (nB_Omega'*nB_Omega);
            dJdb_g = single((1-tanh(U0).^2).*dJdB_g(ix,:));
            dJdoutput = gpuArray(reshape(dJdb_g',[1,1,size(dJdb_g',1),size(dJdb_g',2)])) ;
            res = vl_simplenn( net, im_, dJdoutput);
            net = update_net(net , res, lr((epoch-1)*iters+t), num_gamma,batch_size) ;
            U1 = ones(size(U0));
            U1(tanh(U0) <= 0) = -1;
            nB_Gamma(ix,:) = U1; 
        end
                
        nB(Gamma,:) = nB_Gamma;
        B = nB;
        gamma_time = toc(gamma_time);        

        [loss_, ~, ~] = calc_loss(nB_Gamma, nB_Omega, S_gamma, S_omega, bit);
        loss(count) = loss_;
        count = count + 1;

        fprintf('Iteration: %3d/%3d(%3d), Loss: %.4f, Time: %.4f+%.4f(m)\n', t, iters, (epoch-1)*iters+t, loss_, omega_time / 60, gamma_time / 60);
    end    
end


end

function net = net_structure(net, bit)
net.layers = net.layers(1:19);
n = numel(net.layers); 

for i=1:n
    if isfield(net.layers{i}, 'weights')    
        net.layers{i}.weights{1} = gpuArray(net.layers{i}.weights{1}) ;        
        net.layers{i}.weights{2} = gpuArray(net.layers{i}.weights{2}) ;
    end   
end

net.layers{n+1}.pad = [0,0,0,0];
net.layers{n+1}.stride = [1,1];
net.layers{n+1}.type = 'conv';
net.layers{n+1}.name = 'fc8';
net.layers{n+1}.weights{1} = gpuArray(0.01*randn(1,1,4096,bit,'single'));
net.layers{n+1}.weights{2} = gpuArray(0.01*randn(1,bit,'single'));
net.layers{n+1}.opts = {};
end

function gpu_net = update_net(gpu_net, res_back, lr, numTrain, batch_size)
weightDecay = 5*1e-4 ;
nLayers = numel(gpu_net.layers) ;
for ii = 1:nLayers
    if isfield(gpu_net.layers{ii},'weights')    
        gpu_net.layers{ii}.weights{1} = gpu_net.layers{ii}.weights{1}-...            
            lr*(res_back(ii).dzdw{1}/(batch_size*numTrain) + weightDecay*gpu_net.layers{ii}.weights{1});    
        gpu_net.layers{ii}.weights{2} = gpu_net.layers{ii}.weights{2}-...    
            lr*(res_back(ii).dzdw{2}/(batch_size*numTrain) + weightDecay*gpu_net.layers{ii}.weights{2});    
    end    
end
end

function [loss_, sl1, sl2] = calc_loss(B_g,B_o,S_g,S_o,q)
sl1 = norm(q*S_g-B_g*B_o','fro')^2;
sl2 = norm(q*S_o-B_o*B_o','fro')^2;
loss_ = (sl1 + sl2) / (size(B_g, 1) + size(B_o, 1));
end
