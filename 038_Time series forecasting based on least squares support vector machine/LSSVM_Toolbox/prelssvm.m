function [model, Yt] = prelssvm(model, Xt, Yt)

%%  预处理过程
%
% These functions should only be called by trainlssvm or by
% simlssvm. At first the preprocessing assigns a label to each in-
% and output component (c for continuous, a for categorical or b
% for binary variables). According to this label each dimension is rescaled:
% 
%     * continuous: zero mean and unit variance
%     * categorical: no preprocessing
%     * binary: labels -1 and +1

%%  参数解释
%         [Xp, Yp] = prelssvm(model, Xt, Yt)  
%         model : Preprocessed object oriented representation of the LS-SVM model
%         Xp    : Nt x d matrix with the preprocessed inputs of the test data
%         Yp    : Nt x d matrix with the preprocessed outputs of the test data 
%         model : Object oriented representation of the LS-SVM model
%         Xt    : Nt x d matrix with the inputs of the test data to preprocess
%         Yt    : Nt x d matrix with the outputs of the test data to preprocess

%%  是否预处理
if model.preprocess(1) ~= 'p'
  if nargin >= 2 
     model = Xt;
  end 
end

%%  预处理
if model.preprocess(1) == 'p'
    try
        % 分类处理
        if  model.prestatus(1) == 'c'
            model.prestatus = 'unschemed';
        end

    catch
        model.prestatus = 'unschemed';
    end
end  

if nargin == 1

  %% 处理方式u
  if model.prestatus(1) == 'u'
    % 预分配空间
    ffx = [];
    % 循环处理
    for i = 1 : model.x_dim
        try 
            ffx = [ffx, model.pre_xscheme(i)];
        catch
            ffx = [ffx, signal_type(model.xtrain(:, i), inf)];
        end
    end

    % 重新赋值
    model.pre_xscheme = ffx;
    ff = [];

    for i = 1 : model.y_dim
        try
            ff = [ff, model.pre_yscheme(i)];
        catch
            ff = [ff, signal_type(model.ytrain(:, i), model.type)];
        end
    end

    model.pre_yscheme = ff;
    model.prestatus = 'schemed';

  end
  
  %%  处理方式s（如果尚未编码，则按照定义执行重新缩放）
  if model.prestatus(1) == 's'  
     model = premodel(model); 
     model.prestatus = 'ok';
  end
  
elseif model.preprocess(1) == 'p'
  if model.prestatus(1) == 'o'
      try 
      catch
          Yt = [];
      end
    [model, Yt] = premodel(model, Xt, Yt);

  end
end

%%  确定信号的类型（二分类、多分类和连续）
function [type, ss] = signal_type(signal, type)

ss  = sort(signal);
dif = sum(ss(2 : end) ~= ss(1 : end - 1)) + 1;

% 二分类
if dif == 2
  type = 'b';

% 多分类
elseif (dif < sqrt(length(signal)) || type(1) == 'c')
  type = 'a';

% 连续类型
else
  type = 'c';
end

%%  重新调整模型数据
function [model, Yt] = premodel(model, Xt, Yt)

if nargin == 1
  
  %% 处理输入
  for i = 1 : model.x_dim
    % 连续变量
    if model.pre_xscheme(i) == 'c'
       model.pre_xmean(i) = mean(model.xtrain(:, i));
       model.pre_xstd(i)  = std(model.xtrain(:, i));
       model.xtrain(:, i) = pre_zmuv(model.xtrain(:, i), model.pre_xmean(i), model.pre_xstd(i));
    
    % 多分类变量
    elseif model.pre_xscheme(i) == 'a'
      model.pre_xmean(i) = 0;
      model.pre_xstd(i)  = 0;
      model.xtrain(:, i) = model.xtrain(:, i);
    
    % 二分类变量
    elseif model.pre_xscheme(i) == 'b'
      model.pre_xmean(i) = min(model.xtrain(:, i));
      model.pre_xstd(i)  = max(model.xtrain(:, i));
      model.xtrain(:, i) = pre_bin(model.xtrain(:, i), model.pre_xmean(i));
    end

  end
  
  %%  处理输出
  for i = 1 : model.y_dim
    % 连续变量
    if model.pre_yscheme(i) == 'c'
      model.pre_ymean(i) = mean(model.ytrain(:, i), 1);
      model.pre_ystd(i)  = std(model.ytrain(:, i), 1);
      model.ytrain(:, i) = pre_zmuv(model.ytrain(:, i), model.pre_ymean(i), model.pre_ystd(i));
    
    % 多分类变量 
    elseif model.pre_yscheme(i) == 'a'
      model.pre_ymean(i) = 0;
      model.pre_ystd(i)  = 0;
      model.ytrain(:, i) = model.ytrain(:, i);
    
    % 二分类变量
    elseif model.pre_yscheme(i) == 'b'
      model.pre_ymean(i) = min(model.ytrain(:, i));
      model.pre_ystd(i)  = max(model.ytrain(:, i));
      model.ytrain(:, i) = pre_bin(model.ytrain(:, i), model.pre_ymean(i));
    end

  end

else

  if ~isempty(Xt)
    if size(Xt, 2) ~= model.x_dim
        warning('dimensions of Xt not compatible with dimensions of support vectors...');
    end

    for i = 1 : model.x_dim
      % 连续变量
      if model.pre_xscheme(i) == 'c'
	     Xt(:, i) = pre_zmuv(Xt(:, i), model.pre_xmean(i), model.pre_xstd(i));
      
      % 多分类变量
      elseif model.pre_xscheme(i) == 'a'
          Xt(:, i) = Xt(:, i);

      % 二分类变量
      elseif model.pre_xscheme(i) == 'b'
          Xt(:, i) = pre_bin(Xt(:, i), model.pre_xmean(i));
      end

    end
  end
  
  if nargin > 2 && ~isempty(Yt)

    if size(Yt, 2) ~= model.y_dim 
        warning('dimensions of Yt not compatible with dimensions of training output...');
    end

    for i = 1 : model.y_dim
      
      % 连续变量
      if model.pre_yscheme(i) == 'c'
          Yt(:, i) = pre_zmuv(Yt(:, i), model.pre_ymean(i), model.pre_ystd(i));

      % 多分类变量
      elseif model.pre_yscheme(i) == 'a'
          Yt(:, i) = Yt(:, i);

      % 二分类变量
      elseif model.pre_yscheme(i) == 'b'
          Yt(:,i) = pre_bin(Yt(:, i), model.pre_ymean(i));
      end

    end
  end

  % 得到结果
  model = Xt;
end


%%  预处理连续信号
function X = pre_zmuv(X, mean, var)
X = (X - mean) ./ var;

%%  预处理二进制信号
function X = pre_bin(X, min)
    if ~sum(isnan(X)) >= 1
        n = (X == min);
        p = not(n);
        X = -1 .* n + p;
    end