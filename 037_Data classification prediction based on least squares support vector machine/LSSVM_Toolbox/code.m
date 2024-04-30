function [nsignals, codebook, oldcodebook, scheme] = code(signals, codetype, codetype_args, oldcodebook, ...
    fctdist, fctdist_args)

%%  将多类分类任务编码和解码成多个二元分类器

% Yc = code(Y, codebook)
% The coding is defined by the codebook. The codebook is
% represented by a matrix where the columns represent all different
% classes and the rows indicate the result of the binary
% classifiers. An example is given: the 3 classes with original
% labels [1 2 3] can be encoded in the following codebook (using Minimal Output Encoding):
%
% >> codebook
%     = [-1  -1  1;
%         1  -1  1]
% 
% For this codebook, a member of the first class is found if the
% first binary classifier is negative and the second classifier is
% positive. A don't care is represented by eps. By default it is
% assumed that the original classes are represented as different
% numerical labels. One can overrule this by passing the
% old_codebook which contains information about the old representation.
% 
% Different encoding schemes are available:
% 
%     1. Minimum Output Coding (code_MOC) 
%     2. Error Correcting Output Code (code_ECOC)
%       This coding scheme uses redundant bits. 
%     3. One versus All Coding (code_OneVsAll)
%     4. One Versus One Coding (code_OneVsOns)
% 
% Different decoding schemes are implemented:
% 
%     1. Hamming Distance (codedist_hamming) 
%     2. Bayesian Distance Measure (codedist_bay)

%%  编码
%  [Yc, codebook, old_codebook] = code(Y, codefct)
%         Yc               : N x nbits encoded output classifier
%         codebook(*)      : nbits*nc matrix representing the used encoding
%         old_codebook(*)  : d*nc matrix representing the original encoding
%         Y                : N x d matrix representing the original classifier
%         codefct(*)       : Function to generate a new codebook (e.g. code_MOC)

%%  解码
% >> Yd = code(Yc, codebook, [], old_codebook)
% >> Yd = code(Yc, codebook, [], old_codebook, codedist_fct)
% >> Yd = code(Yc, codebook, [], old_codebook, codedist_fct, codedist_args)

%         Yd               : N x nc decoded output classifier  
%         Y                : N x d matrix representing the original classifier
%         codebook         : d*nc matrix representing the original encoding
%         old_codebook     : bits*nc matrix representing the encoding of the given classifier
%         codedist_fct     : Function to calculate the distance between to encoded classifiers (e.g. codedist_hamming)

%%  默认处理
try
    fctdist(1, 1);
catch
    fctdist = 'codedist_hamming';
end

try
    if isempty(oldcodebook)
       ss = sort(signals(:, 1));  
       oldcodebook = ss([1; find(ss(2 : end) ~= ss(1 : end - 1)) + 1])';
    end
catch
    ss = sort(signals(:, 1));  
    oldcodebook = ss([1; find(ss(2 : end)~=ss(1 : end - 1)) + 1])';
end

%%  参数设置
n = size(signals, 1);
mc = size(oldcodebook, 2);

% 码本或码型，初始化用于预处理的新方案
if isstr(codetype)
    try
        [codebook, scheme] = feval(codetype, mc, codetype_args{:});
    catch
        [codebook, scheme] = feval(codetype, mc);
    end
else
  codebook = codetype;
  scheme = []; 
  for t = 1 : size(codebook, 2) 
      scheme = [scheme, 'b']; 
  end
end


% 从旧编码转换为新编码
if nargin == 6
  dist = feval(fctdist, signals, oldcodebook, fctdist_args{:});
else
  dist = feval(fctdist, signals, oldcodebook);
end

for t = 1:n
    [~, mi] = min(dist(t, :));
    nsignals(t,: ) = codebook(:, mi)';
end