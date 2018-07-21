function p = gaussian_pos_prob(X, Mu, Sigma, Phi)
%GAUSSIAN_POS_PROB Posterior probability of GDA.
%   p = GAUSSIAN_POS_PROB(X, Mu, Sigma) compute the posterior probability
%   of given N data points X using Gaussian Discriminant Analysis where the
%   K gaussian distributions are specified by Mu, Sigma and Phi.
%
%   Inputs:
%       'X'     - M-by-N matrix, N data points of dimension M.
%       'Mu'    - M-by-K matrix, mean of K Gaussian distributions.
%       'Sigma' - M-by-M-by-K matrix (yes, a 3D matrix), variance matrix of
%                   K Gaussian distributions.
%       'Phi'   - 1-by-K matrix, prior of K Gaussian distributions.
%
%   Outputs:
%       'p'     - N-by-K matrix, posterior probability of N data points
%                   with in K Gaussian distributions.

N = size(X, 2);
K = length(Phi);
p = zeros(N, K);

% Your code HERE
py = Phi;
plike = zeros(1, K);
for n = 1:N
    px = 0;
    for k = 1:K
        plike(1,k) = 1/(2*pi*sqrt(det(Sigma(:, :, k)))) * exp(-1/2 * (X(:, n)-Mu(:,k))' * inv(Sigma(:, :, k)) * (X(:, n)-Mu(:,k)));   
        px = px + plike(1, k) * py(1, k);
    end
    p(n, :) = plike .* py / px;
end