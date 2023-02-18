function [val, W1, B1, W2, B2] = gadecod(x)

%%  读取主空间变量
S1 = evalin('base', 'S1');             % 读取隐藏层神经元个数
net = evalin('base', 'net');           % 读取网络参数
p_train = evalin('base', 'p_train');   % 读取输入数据
t_train = evalin('base', 't_train');   % 读取输出数据

%%  参数初始化
R2 = size(p_train, 1);                 % 输入节点数 
S2 = size(t_train, 1);                 % 输出节点数

%%  输入权重编码
for i = 1 : S1
    for k = 1 : R2
        W1(i, k) = x(R2 * (i - 1) + k);
    end
end

%%  输出权重编码
for i = 1 : S2
    for k = 1 : S1
        W2(i, k) = x(S1 * (i - 1) + k + R2 * S1);
    end
end

%%  隐层偏置编码
for i = 1 : S1
    B1(i, 1) = x((R2 * S1 + S1 * S2) + i);
end

%%  输出偏置编码
for i = 1 : S2
    B2(i, 1) = x((R2 * S1 + S1 * S2 + S1) + i);
end

%%  赋值并计算
net.IW{1, 1} = W1;
net.LW{2, 1} = W2;
net.b{1}     = B1;
net.b{2}     = B2;

%%  模型训练
net.trainParam.showWindow = 0;      % 关闭训练窗口
net = train(net, p_train, t_train);

%%  仿真测试
t_sim1 = sim(net, p_train);

%%  计算适应度值
val =  1 ./ (sqrt(sum((t_sim1 - t_train).^2) ./ length(t_sim1)));
