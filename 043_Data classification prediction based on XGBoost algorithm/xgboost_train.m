function model = xgboost_train(p_train, t_train, params, max_num_iters)
%%% Function inputs:
% p_train:        matrix of inputs for the training set
% t_train:        vetor of labels/values for the test set
% params :        structure of learning parameters
% max_num_iters: max number of iterations for learning

%%% Function output:
% model: a structure containing:
%     iters_optimal; % number of iterations performs by xgboost (final model)
%     h_booster_ptr; % pointer to the final model
%     params;        % model parameters (just for info)
%     missing;       % value considered "missing"

%%  加载 xgboost 库
loadlibrary('xgboost')

%%  设置参数
missing = single(NaN);          % 设置该值被视为"缺失"
iters_optimal = max_num_iters;  % 最大迭代次数

%%  设置xgboost的相关参数
if isempty(params)
    params.booster           = 'gbtree';
    % params.objective         = 'binary:logistic';
    params.objective         = 'reg:linear';
    params.max_depth         = 5;
    params.eta               = 0.1;
    params.min_child_weight  = 1;
    params.subsample         = 0.9;
    params.colsample_bytree  = 1;
    params.num_parallel_tree = 1;
end

%%  将属性转换为全局属性
param_fields = fields(params);
for i = 1 : length(param_fields)
    eval(['params.' param_fields{i} ' = num2str(params.' param_fields{i} ');'])
end

%%  得到输入数据相关属性
rows = uint64(size(p_train, 1));  % 输入数据的行 样本数
cols = uint64(size(p_train, 2));  % 输入数据的列 特征数
p_train = p_train';

%%  创建相关指针
p_train_ptr = libpointer('singlePtr', single(p_train));
t_train_ptr = libpointer('singlePtr', single(t_train));

h_train_ptr = libpointer;
h_train_ptr_ptr = libpointer('voidPtrPtr', h_train_ptr);

%%  处理输入特征
calllib('xgboost', 'XGDMatrixCreateFromMat', p_train_ptr, rows, cols, missing, h_train_ptr_ptr);

%%  处理标签
labelStr = 'label';
calllib('xgboost', 'XGDMatrixSetFloatInfo', h_train_ptr, labelStr, t_train_ptr, rows);

%%  建立集成器并设置参数
h_booster_ptr = libpointer;
h_booster_ptr_ptr = libpointer('voidPtrPtr', h_booster_ptr);
calllib('xgboost', 'XGBoosterCreate', h_train_ptr_ptr, uint64(1), h_booster_ptr_ptr);

for i = 1 : length(param_fields)
    eval(['calllib(''xgboost'', ''XGBoosterSetParam'', h_booster_ptr, ''' param_fields{i} ''', ''' eval(['params.' param_fields{i}]) ''');'])
end

%%  最终模型
for iter = 0 : iters_optimal
    calllib('xgboost', 'XGBoosterUpdateOneIter', h_booster_ptr, int32(iter), h_train_ptr);
end

%%  将模型参数保存到 model
model                = struct;
model.iters_optimal  = iters_optimal;  % 最大迭代次数
model.h_booster_ptr  = h_booster_ptr;  % 指向最终模型的指针
model.params         = params;         % 相关参数
model.missing        = missing;        % 缺失值
