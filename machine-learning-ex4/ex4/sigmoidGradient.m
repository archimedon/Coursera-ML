function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).



% In the lecture:
%	(@ 5:00) <https://www.coursera.org/learn/machine-learning/lecture/1z9WW/backpropagation-algorithm?t=300>
% Ng says the derivative of sigmoid(z[l]) written, g'(z[l]) is: a[l] .* (1 - a[l]) ;
%		where"
%		a[l] = sigmoid(z[l])

% g(z).*(1−g(z))
a = sigmoid(z);
g = a .* (1 - a);

%



% =============================================================




end
