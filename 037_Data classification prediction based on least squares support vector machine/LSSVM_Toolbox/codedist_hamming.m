function dist = codedist_hamming(C1, C2)

%% 计算'C1'的行和'C2'的列之间的 hamming 距离
% distance = codedist_hamming(encoded_data, codebook);
% 
% 'encoded_data' contains the resulting codeword per row, n rows are possible
% 'codebook' contains the codebooks prototype per class as columns.
%  an infinitesimal number 'eps'  represents the don't care

%%  参数设置
nb = size(C1, 1);
[nbin, dim] = size(C2);
dist = zeros(nb, dim);

%%  计算距离
for d = 1 : dim
  for n= 1 : nb
    dist(n, d) = nbin - sum(C1(n, :) == C2(:, d)' | C1(n, :) < -10000 | ...
        C1(n, :) == eps | C1(n, :) > 10000 | C2(:, dim)' == eps);
  end
end