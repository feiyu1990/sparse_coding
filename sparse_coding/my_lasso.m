function [ S ] = my_lasso( X, B, lambda )
%MY_LASSO Summary of this function goes here
%   Detailed explanation goes her

[n, p] = size(B);
M= size(X, 2);
S = zeros(p, M);
for i=1:M,
    S(:,i) = lasso(B, X(:,1), 'Lambda', lambda);
    disp(i);
end;
end

