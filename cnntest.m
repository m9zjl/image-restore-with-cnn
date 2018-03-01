function [er, bad] = cnntest(net, x, y)
    %  feedforward
    net = cnnff_test(net, x);  % 前向传播得到输出
 %   figure(2);
    net.o=reshape(net.o',128,128)*255;
 %   imshow(net.o);
    y=reshape(y,128,128,5);
 %   figure(3);
 %   imshow(y(:,:,3))
    n = size(y,3);
    for k = 1:n
        sum = 0;
        Y = y(:,:,k);
        for i = 1:128
            for j= 1:128
                sum = (net.o(i,j) - Y(i,j))^2 + sum;
                A(k) = sum
            end
        end 
    end
    [m,t]=min(A);
    a = t
    y = uint8(y(:,:,t));
    if (m<3e7)
        {
            figure(2);
            imshow(y);
         }   
    else
        disp('Error');
    end
    
    % [Y,I] = max(X) returns the indices of the maximum values in vector I 
    [~, h] = max(net.o);  % 找到最大的输出对应的标签
    [~, a] = max(y);      % 找到最大的期望输出对应的索引
    bad = find(h ~= a);   % 找到他们不相同的个数，也就是错误的次数

    er = numel(bad) / size(y, 2);  % 计算错误率  
end
