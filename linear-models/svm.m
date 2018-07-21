function [w, num] = svm(X, y)
%SVM Support vector machine.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           num:  number of support vectors
%

% YOUR CODE HERE
X = [ones(1,size(X, 2));X];
w = rand(size(X, 1), 1);
H = eye(size(X, 1));
b = ones(size(X, 2), 1);
f = zeros(1, size(X, 1));
b = -b;
ytemp = zeros(size(X, 2) - 1);
for i = 1:size(X, 2)
   ytemp(i,i) = y(1,i); 
end
A = (X * ytemp)';
w = quadprog(H, f, A, b);
end
