function [model, Yt] = postlssvm(model, Xt, Yt)

%%  后期处理
%
% These functions should only be called by trainlssvm or by
% simlssvm. At first the preprocessing assigns a label to each in-
% and output component (c for continuous, a for categorical or b
% for binary variables). According to this label each dimension is rescaled:
% 
%     * continuous: zero mean and unit variance
%     * categorical: no preprocessing
%     * binary: labels -1 and +1

%         [Xp, Yp] = postlssvm(model, Xt, Yt)
%         model : Preprocessed object oriented representation of the LS-SVM model
%         Xt    : Nt x d matrix with the inputs of the test data to preprocess
%         Yt    : Nt x d matrix with the outputs of the test data to preprocess
%         model : Object oriented representation of the LS-SVM model
%         Xp    : Nt x d matrix with the preprocessed inputs of the test data
%         Yp    : Nt x d matrix with the preprocessed outputs of the test data

%
%%  后期测试处理

%%  解码
if model.preprocess(1) ~= 'p'
  if nargin >= 2
    % 无需后处理，无需编码
    model = Xt; 
  end
  return
end


%%  后处理LSSVM
if nargin == 1
    % 按照定义执行重新缩放
    if (model.prestatus(1)=='o' && model.preprocess(1)=='p') || ... 
       (model.prestatus(1)=='c' && model.preprocess(1)=='o')

      model = postmodel(model);   
      model.preprocess = 'original'; 

    end

    model.prestatus='ok';

% 重新缩放以模拟输入
else
    try
    catch
        Yt = [];
    end
  [model, Yt] = postmodel(model, Xt, Yt);
end

%%  后期处理模型
function [model, Yt] = postmodel(model, Xt, Yt)

if nargin == 1
  
  for i = 1 : model.x_dim
    % 连续变量
    if model.pre_xscheme(i) == 'c'
       model.xtrain(:, i) = post_zmuv(model.xtrain(:, i), model.pre_xmean(i), model.pre_xstd(i));

    % 多分类变量
    elseif model.pre_xscheme(i) == 'a'
      model.xtrain(:, i) = model.xtrain(:, i);

    % 二分类变量 
    elseif model.pre_xscheme(i) == 'b'
      model.xtrain(:, i) = post_bin(model.xtrain(:, i), model.pre_xmean(i), model.pre_xstd(i));
    end

  end

  %%  处理输出
  for i = 1 : model.y_dim
    
    % 连续变量
    if model.pre_yscheme(i) == 'c'
      model.ytrain(:, i) = post_zmuv(model.ytrain(:, i), model.pre_ymean(i), model.pre_ystd(i));

    % 多分类变量
    elseif model.pre_yscheme(i) == 'a'   
      model.ytrain(:, i) = model.ytrain(:, i);

    % 二分类变量
    elseif model.pre_yscheme(i) == 'b'
      model.ytrain(:, i) = post_bin(model.ytrain(:, i), model.pre_ymean(i), model.pre_ystd(i));
    end  
  end

else
  
  %% 
  if nargin > 1
      if ~isempty(Xt)
          for i = 1 : model.x_dim
              % 连续变量
              if model.pre_xscheme(i) == 'c'
	              Xt(:,i) = post_zmuv(Xt(:,i),model.pre_xmean(i),model.pre_xstd(i));

              % 多分类变量
              elseif model.pre_xscheme(i)=='a'
	              Xt(:,i) = Xt(:,i);

	          % 二分类变量
	          elseif model.pre_xscheme(i)=='b'
	              Xt(:,i) = post_bin(Xt(:,i),model.pre_xmean(i),model.pre_xstd(i));
              end
          end
      end

    %%  
    if nargin > 2 && ~isempty(Yt)
      for i = 1 : model.y_dim

	    % 连续变量
	    if model.pre_yscheme(i) == 'c'
	      Yt(:, i) = post_zmuv(Yt(:, i), model.pre_ymean(i), model.pre_ystd(i));

	    % 多分类变量
	    elseif model.pre_yscheme(i) == 'a'     
	      Yt(:, i) = Yt(:, i);

	    % 二分类变量 
	    elseif model.pre_yscheme(i) == 'b'
	      Yt(:, i) = post_bin(Yt(:, i), model.pre_ymean(i), model.pre_ystd(i));
	    end
      end
    end
    
    model = Xt;
  end
end

%%  连续变量处理函数
function X = post_zmuv(X, mean, var)
X = X .* var + mean;

%%  二分类处理函数
function X = post_bin(X, min, max)
X = min .* (X <= 0) + max .* (X > 0);
