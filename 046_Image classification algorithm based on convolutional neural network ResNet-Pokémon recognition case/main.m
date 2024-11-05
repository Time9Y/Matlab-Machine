warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
imds = imageDatastore('images', ...       % 读取文件夹名称
        'IncludeSubfolders', true, ...    % 是否包含子文件夹 
        'LabelSource', 'foldernames');    % 将子文件夹名作为标签
    
%%  划分数据
[imdTrain, imdTest] = splitEachLabel(imds, 0.8, 'randomized');

%%  获取类别数目
numClasses = numel(categories(imdTrain.Labels));

%%  加载预训练网络
net = resnet18;
analyzeNetwork(net)
img_size = net.Layers(1).InputSize(1: 2);

%% 获取网络的层并修改最后的全连接层和分类层
lgraph = layerGraph(net);

% 查找原始的全连接层和分类层
fcLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc');
classificationLayer = classificationLayer('Name', 'new_classoutput');

% 替换原来的全连接层和分类层
lgraph = replaceLayer(lgraph, 'fc1000', fcLayer);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', classificationLayer);

%%  数据增强
pixelRange = [-10, 10];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation'    , pixelRange,  ...    % 旋转角度范围
    'RandXReflection' , true,        ...    % 上下方向的随机反射
    'RandXTranslation', pixelRange,  ...    % 水平平移范围
    'RandYTranslation', pixelRange);        % 垂直平移范围

Train = augmentedImageDatastore(img_size, imdTrain, 'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');
Test  = augmentedImageDatastore(img_size, imdTest , 'ColorPreprocessing', 'gray2rgb');

%%  参数设置
options = trainingOptions('adam', ...      % Adam 梯度下降算法
    'MiniBatchSize', 64, ...               % 批大小, 每次训练样本个数
    'MaxEpochs', 30, ...                   % 最大训练次数
    'InitialLearnRate', 1e-3, ...          % 初始学习率为
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.1, ...        % 学习率下降因子
    'LearnRateDropPeriod', 20, ...         % 每经过20次训练, 学习率 = 学习率 * 下降因子
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'ValidationData', Test, ...            % 验证数据集
    'ValidationFrequency', 20, ...         % 每20进行一次验证
    'Plots', 'training-progress',...       % 绘制损失曲线
    'Verbose', false);                     % 关闭命令行显示

%%  模型训练
net = trainNetwork(Train, lgraph, options);

%%  仿真测试
T_sim1 = classify(net, Train);
T_sim2 = classify(net, Test );

%%  设置输出
T_train = imdTrain.Labels;
T_test  = imdTest.Labels ;

%%  性能评价
accuracy1 = mean(T_sim1 == T_train) * 100;
accuracy2 = mean(T_sim2 == T_test ) * 100;

%%  显示准确率
disp(['训练集准确率：', num2str(accuracy1), '%'] )
disp(['测试集准确率：', num2str(accuracy2), '%'] )

%%  混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

%%  保存网络
save net.mat net