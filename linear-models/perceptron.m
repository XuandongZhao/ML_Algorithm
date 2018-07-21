function [w, iter] = perceptron(X, y)
%PERCEPTRON Perceptron Learning Algorithm.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           iter: number of iterations
%

% YOUR CODE HERE
% train
[P, N] = size(X);
w = rand(P+1, 1);
maxiter = 2000;
iter = 0;
alpha = 0.01;
X = [ones(1,N);X];
while iter < maxiter
    err = 0;
    d_w = zeros(P+1, 1);
    y_cal = sign((w') * X .* y);
    for i = 1:N
        if(y_cal(1,i)<0)
            err = 1;
            d_w = d_w + X(:,i) * y(1,i);
        end
    end
    if(err == 0)
        break;
    end
    w = w + alpha * d_w;
    iter = iter + 1;
end
