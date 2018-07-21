function w = logistic(X, y)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% % YOUR CODE HERE
[P, N] = size(X);
X = [ones(1, N); X];
w = rand(P+1, 1);
lr = 0.1;
d_w = zeros(size(X,1),1);
for iter = 1:100
    for i = 1:size(X,2)
        d_w = d_w + y(1,i) * X(:,i) * 1.0 / (1.0 + exp(y(1,i) .* (w') * X(:,i)));
    end
    w = w + lr * (1.0 / iter) * d_w;
end

end
