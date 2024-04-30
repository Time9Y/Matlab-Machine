function [codebook, scheme] = code_OneVsAll(m)

%% 使用 One-Versus-All 编码生成用于多类分类的码本

codebook = eye(m) .* 2 - 1;

% 输出格式
scheme = []; 
for i = 1 : m 
    scheme = [scheme, 'b']; 
end