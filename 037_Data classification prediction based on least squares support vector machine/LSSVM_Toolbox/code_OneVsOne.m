function [codebook, scheme] = code_OneVsOne(m)

%% 使用一对一编码生成用于多类分类的码本

% 参数设置
nb = m * (m - 1) / 2;
codebook = NaN * zeros(nb, m);

% 编码
t = 1;
for i = 1 : m - 1
  for j = i + 1 : m
       codebook(t, i) =  1;
       codebook(t, j) = -1;
       t = t + 1;
  end
end

% 输出格式
scheme = []; 
for i = 1 : nb
    scheme = [scheme, 'b']; 
end