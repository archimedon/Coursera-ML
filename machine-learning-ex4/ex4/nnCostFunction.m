function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
% 
% warning: operator -: automatic broadcasting operation applied
% size Theta1_hyp: (5000, 25)
% size Theta2_hyp: (5000, 10)
% size zeroOne: (10, 26)
% size offset: (26, 10)
% size regulator: (26, 10)
% size Theta2_grad: (26, 10)

fprintf("size num_labels: (%d, %d)\n", size(num_labels));

fprintf("size J: (%d, %d)\n", size(J));
fprintf("size X: (%d, %d)\n", size(X));
fprintf("size Theta1: (%d, %d)\n", size(Theta1));
fprintf("size Theta1_hyp: (%d, %d)\n", size(Theta1_hyp));
fprintf("size Theta2: (%d, %d)\n", size(Theta2));
fprintf("size y: (%d, %d)\n", size(y));
fprintf("size zeroOne: (%d, %d)\n", size(zeroOne));
fprintf("size offset: (%d, %d)\n", size(offset));
fprintf("size regulator: (%d, %d)\n", size(regulator));
fprintf("size Theta1_grad: (%d, %d)\n", size(Theta1_grad));



zeroOne = noTheta0(Theta2);
Theta2_hyp = sigmoid( pad(Theta1_hyp) * zeroOne');
offset = (1/m * (pad(Theta1_hyp)' * (Theta2_hyp - y))) ;
regulator = (lambda/m) * zeroOne';
Theta2_grad = offset + regulator;



fprintf("size Theta1_hyp: (%d, %d)\n", size(Theta1_hyp));
fprintf("size Theta2_hyp: (%d, %d)\n", size(Theta2_hyp));
fprintf("size zeroOne: (%d, %d)\n", size(zeroOne));
fprintf("size offset: (%d, %d)\n", size(offset));
fprintf("size regulator: (%d, %d)\n", size(regulator));
fprintf("size Theta2_grad: (%d, %d)\n", size(Theta2_grad));

% size X: (5000, 400)
% size Theta1: (25, 401)
% size Theta1_hyp: (5000, 25)
% size Theta2: (10, 26)
% size y: (5000, 1)
% size zeroOne: (25, 401)
% size offset: (401, 25)
% size regulator: (401, 25)
% size Theta1_grad: (401, 25)


initial_theta = zeros(n + 1, 1);

options = optimset('GradObj', 'on', 'MaxIter', 50);

for c = 1:num_labels,
	% Create a boolean vector that denotes the value for class 'c'
	y_vector = (y==c);
	[theta] = fmincg(@(t)(lrCostFunction(t, X, y_vector, lambda)), initial_theta, options);
	all_theta(c, :) = theta';
end



%% set elem(1) to zero
function theta = noTheta0(theta)
	theta(1) = 0;
end




% [hyp_max, hyp_index] .
% Set 'p' to column-index of max hypThetaOfX indicating the classifier with the strongest assertion.
% column-index is equal to class
% [hyp_max, p] = max(hyps, [], 2);

function hyps = calc(inputs, theta),
	[m, n] = size(inputs);
	inputs = [ones(m, 1), inputs];
	hyps = sigmoid(inputs * theta');
end

function matplus1 = pad(matrix),
	[m, n] = size(matrix);
	matplus1 = [ones(m, 1), matrix];
end














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
