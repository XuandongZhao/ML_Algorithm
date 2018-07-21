%% Ridge Regression
load('digit_train', 'X', 'y');

% Do feature normalization
Xnorm = zeros(size(X));
for i = 1:size(X, 2)
    Xnorm(:, i) = (X(:, i) - mean(X(:, i))) / std(X(:, i));
end
X = Xnorm;

% Do LOOCV
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
lambda = 0;
E_vals = zeros(length(lambdas), 1);
for i = 1:length(lambdas)
    E_val = 0;
    loss = 0;
    for j = 1:size(X, 2)
        X_ = X;
        X_(:,j) = [];
        y_ = y; 
        y_(:,j) = [];
        w = ridge(X_, y_, lambdas(i));
        E_val = E_val + (sign(w' * [1; X(:, j)]) ~= y(j));
        X_ = [ones(1, size(X_,2)); X_];
        loss = loss + (y_' - X_' * w)' * (y_' - X_' * w) + lambdas(i) * (w' * w);
    end
    loss = loss / size(X,2);
    E_val = E_val / size(X,2);
    fprintf('labmda = %f  Error rate = %f  Loss = %f\n', lambdas(i), E_val, loss);
    % Update lambda according validation error
    E_vals(i) = E_val;
end
[~, I] = sort(E_vals);
lambda = lambdas(I(1));
fprintf('The best lambda: %f\n', lambda);
% Compute training error
w = ridge(X, y, 0);
w_r = ridge(X, y, lambda);
fprintf('The sum of omega square with regularization: %f\n', sum(w_r.^2));
fprintf('The sum of omega square without regularization: %f\n', sum(w.^2));
E_train = sum(sign(w' * [ones(1, size(X, 2)); X]) ~= y) / size(y, 2);
E_train_r = sum(sign(w_r' * [ones(1, size(X, 2)); X]) ~= y) / size(y, 2);
fprintf('Without regularization, the train error is %f.\n', E_train);
fprintf('With regularization, the train error is %f.\n', E_train_r);

load('digit_test', 'X_test', 'y_test');
% Do feature normalization
X_tnorm = zeros(size(X_test));
for i = 1:size(X_test, 2)
    X_tnorm(:, i) = (X_test(:, i) - mean(X_test(:, i))) / std(X_test(:, i));
end
X_test = X_tnorm;
% Compute test error
E_test = sum(sign(w' * [ones(1, size(X_test, 2)); X_test]) ~= y_test) / size(y_test, 2);
E_test_r = sum(sign(w_r' * [ones(1, size(X_test, 2)); X_test]) ~= y_test) / size(y_test, 2);
fprintf('Without regularization, the test error is %f.\n', E_test);
fprintf('With regularization, the test error is %f.\n', E_test_r);

%% Logistic
load('digit_train', 'X', 'y');
Xnorm = zeros(size(X));
for i = 1:size(X, 2)
    Xnorm(:, i) = (X(:, i) - mean(X(:, i))) / std(X(:, i));
end
X = Xnorm;
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
lambda = 0;
E_vals = zeros(length(lambdas), 1);
for i = 1:length(lambdas)
    E_val = 0;
    loss = 0;
    for j = 1:size(X, 2)
        X_ = X;
        X_(:,j) = [];
        y_ = y; 
        y_(:,j) = [];
        w = logistic_r(X_, y_, lambdas(i));
        E_val = E_val + (sign(w' * [1; X(:, j)]) ~= y(j));
        X_ = [ones(1, size(X_,2)); X_];
        loss = loss + (y_' - X_' * w)' * (y_' - X_' * w) + lambdas(i) * (w' * w);
    end
    loss = loss / size(X,2);
    E_val = E_val / size(X,2);
    fprintf('labmda = %f  Error rate = %f  Loss = %f\n', lambdas(i), E_val, loss);
    % Update lambda according validation error
    E_vals(i) = E_val;
end
[~, I] = sort(E_vals);
lambda = lambdas(I(1));
fprintf('The best lambda: %f\n', lambda);
% Compute training error
w = logistic_r(X, y, 0);
w_r = logistic_r(X, y, lambda);
fprintf('The sum of omega square with regularization: %f\n', sum(w_r.^2));
fprintf('The sum of omega square without regularization: %f\n', sum(w.^2));
E_train = sum(sign(w' * [ones(1, size(X, 2)); X]) ~= y) / size(y, 2);
E_train_r = sum(sign(w_r' * [ones(1, size(X, 2)); X]) ~= y) / size(y, 2);
fprintf('Without regularization, the train error is %f.\n', E_train);
fprintf('With regularization, the train error is %f.\n', E_train_r);

load('digit_test', 'X_test', 'y_test');
% Do feature normalization
X_tnorm = zeros(size(X_test));
for i = 1:size(X_test, 2)
    X_tnorm(:, i) = (X_test(:, i) - mean(X_test(:, i))) / std(X_test(:, i));
end
X_test = X_tnorm;
% Compute test error
E_test = sum(sign(w' * [ones(1, size(X_test, 2)); X_test]) ~= y_test) / size(y_test, 2);
E_test_r = sum(sign(w_r' * [ones(1, size(X_test, 2)); X_test]) ~= y_test) / size(y_test, 2);
fprintf('Without regularization, the test error is %f.\n', E_test);
fprintf('With regularization, the test error is %f.\n', E_test_r);
%% SVM with slack variable
