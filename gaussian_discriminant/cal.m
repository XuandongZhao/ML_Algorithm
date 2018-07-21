mu0 =[0;0];
Sigma0 =[1,0;0,3];
mu1 =[0;0];
Sigma1 =[1, 0; 0, 1];
phi =0.5;
Sigma0' * Sigma0
A = inv(Sigma0) - inv(Sigma1);
b = 2*(inv(Sigma1)*mu1 - inv(Sigma0)*mu0);
c = mu0' * inv(Sigma0) * mu0 - mu1' * inv(Sigma1) * mu1 + log(det(Sigma0)/det(Sigma1));
A
b
c
