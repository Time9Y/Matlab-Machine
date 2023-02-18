function change = delta(ct, mt, y, b)

% delta 函数是非均匀突变使用的非均匀分布。
% 此函数根据当前发电量、最大发电量和可能的偏差量返回变化。
%
% ct - current generation
% mt - maximum generation
% y  - maximum amount of change, i.e. distance from parameter value to bounds
% b  - shape parameter

%%  
r = ct / mt;
if(r > 1)
  r = 0.99;
end
change = y * (rand * (1 - r)) ^ b;