function [dataset, param] = load_data(dataname)
switch dataname
    case 'CIFAR-10'
        load ./data/CIFAR-10/CIFAR-10.mat XTrain XTest XDatabase testL trainL databaseL;
        dataset.XTrain = XTrain;
        dataset.XTest = XTest;
        dataset.XDatabase = XDatabase;
        dataset.trainL = createOneHot(double(trainL));
        dataset.testL = double(testL);
        dataset.databaseL = double(databaseL);
                
        param.sample_column = 100;
        param.epochs = 50;
        param.iters = 3;
        
        param.batch_size = 128;
        param.lr = logspace(-2, -4, param.epochs * param.iters);
end
end