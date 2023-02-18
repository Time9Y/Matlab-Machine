function x = parse(inStr)

% parse 是一个函数，它接收一个由空格分隔的文本组成的字符串向量，
% 并将各个字符串项解析为一个 n 项矩阵，每个字符串一行。

% x     - the return matrix of strings
% inStr - the blank separated string vector

%%  切割字符串
strLen = size(inStr, 2);
x = blanks(strLen);
wordCount = 1;
last = 0;
for i = 1 : strLen
  if inStr(i) == ' '
    wordCount = wordCount + 1;
    x(wordCount, :) = blanks(strLen);
    last = i;
  else
    x(wordCount, i - last) = inStr(i);
  end
end