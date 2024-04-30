function [codebook, scheme] = code_MOC(m)

%%  使用最小输出编码生成用于多类分类的码本

% 参数设置
nb = ceil(log2(m));
codebook = -ones(nb, m);

% 编码
for i = 1 : m
  code = str2num(num2str(dec2bin(i - 1)')) .* 2 - 1;
  codebook((nb - length(code) + 1) : nb, i) = code;
end

% 输出格式
scheme = []; 
for i=1 : nb
    scheme = [scheme, 'b']; 
end