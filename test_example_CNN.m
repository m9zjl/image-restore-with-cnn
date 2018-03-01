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


opts.alpha = 0.03;            % ѧϰ�� 
opts.batchsize = 2;        % ÿ������һ��batchsize��batch��ѵ����Ҳ����ÿ��batchsize�������͵���һ��Ȩֵ��������
                           % �����������������ˣ�������������������˲ŵ���һ��Ȩֵ  
opts.numepochs = 1;        % ѵ����������ͬ������������1��ʱ�� 11.41% error��5��ʱ�� 4.2% error��10��ʱ�� 2.73% error 

% �����cnn�����ø�cnnsetup������ݴ˹���һ��������CNN���磬������
cnn = cnnsetup(cnn, train_x, train_y);
% Ȼ��ʼ��ѵ��������������ʼѵ�����CNN����
cnn = cnntrain(cnn, train_x, train_y, opts);
% Ȼ����ò�������������
[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
%figure(3); plot(cnn.rL);
%assert(er<0.12, 'Too big error');
