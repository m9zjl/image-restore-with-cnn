%function test_example_CNN
load mnist_uint8_rec1;

train_x = double(reshape(train_x',128,128,40))/255;
test_x = double(reshape(test_x',128,128,1))/255;
train_y = double(train_y');
test_y = double(test_y');
figure(1);
imshow(test_x);

%% ex1 Train 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};


opts.alpha = 0.03;            % 学习率 
opts.batchsize = 2;        % 每次挑出一个batchsize的batch来训练，也就是每用batchsize个样本就调整一次权值，而不是
                           % 把所有样本都输入了，计算所有样本的误差了才调整一次权值  
opts.numepochs = 1;        % 训练次数，用同样的样本集。1的时候 11.41% error；5的时候 4.2% error；10的时候 2.73% error 

% 这里把cnn的设置给cnnsetup，它会据此构建一个完整的CNN网络，并返回
cnn = cnnsetup(cnn, train_x, train_y);
% 然后开始把训练样本给它，开始训练这个CNN网络
cnn = cnntrain(cnn, train_x, train_y, opts);
% 然后就用测试样本来测试
[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
%figure(3); plot(cnn.rL);
%assert(er<0.12, 'Too big error');
