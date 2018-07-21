function s = sigmoid(x)

s = zeros(size(x));
s = 1./(1+exp(-x));

end