function [model, H] = lssvmMATLAB(model) 
%%  LSSVM算法

%%  计算 omega 和 H
omega = kernel_matrix(model.xtrain(model.selector, 1 : model.x_dim), ...
    model.kernel_type, model.kernel_pars);


%%  初始化参数
model.b = zeros(1, model.y_dim);
model.alpha = zeros(model.nb_data, model.y_dim);

%%  迭代运算得到权重参数
for i = 1 : model.y_dim

    H = omega;
    model.selector = ~isnan(model.ytrain(:, i));
    nb_data = sum(model.selector);

    if size(model.gam, 2) == model.nb_data

      try 
          invgam = model.gam(i, :) .^ -1; 
      catch 
          invgam = model.gam(1, :) .^ -1;
      end

      for t = 1 : model.nb_data
          H(t, t) = H(t, t) + invgam(t);
      end

    else

      try 
          invgam = model.gam(i, 1) .^ -1; 
      catch 
          invgam = model.gam(1, 1) .^ -1;
      end

      for t = 1 : model.nb_data
          H(t, t) = H(t, t) + invgam; 
      end

    end    

    v  = H(model.selector, model.selector) \ model.ytrain(model.selector, i);
    nu = H(model.selector, model.selector) \ ones(nb_data, 1);
    s  = ones(1, nb_data) * nu(:, 1);

    model.b(i) = (nu(:, 1)' * model.ytrain(model.selector, i)) ./ s;
    model.alpha(model.selector, i) = v(:, 1) - (nu(:, 1) * model.b(i));
    
end