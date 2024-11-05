%%  获取网络参数
load net.mat

%%  模型参数
img_size = net.Layers(1).InputSize(1: 2);

%%  读取图像
[filename, pathname] = uigetfile('*', '选择一张图片');
IMG = imread([pathname, filename]);

%%  图像裁剪
I = double(imresize(IMG, img_size));

%%  图像分类
T_sim = classify(net, I);

%% 打印结果
imshow(IMG)
title(T_sim)