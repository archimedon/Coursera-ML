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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% size X: 5000 x 400
% size y: 5000 x 1
% size nn_params: 10285 x 1
% size input_layer_size: 400.000000
% size hidden_layer_size: 25.000000
% size lambda: 0.000000
% size J: 0.000000
% size Theta1: 25 x 401
% size Theta2: 10 x 26
% size Theta1_grad: 25 x 401
% size Theta2_grad: 10 x 26

a1 = prependCol(1, X);
% fprintf('size a1 (expected: 5000x401) : %s' , sizzer(a1));

% (m x n) X (n x h) --> (m x h)
z2 = a1 * Theta1';
% fprintf('size z2 (expected: 5000x25) : %s' , sizzer(z2));

a2 = sigmoid(z2);
% fprintf('size a2 (expected: 5000x25) : %s' , sizzer(a2));

a2 = prependCol(1, a2);
% fprintf('size a2 (expected: 5000x26) : %s' , sizzer(a2));

z3 = a2 * Theta2';
% fprintf('size z3 (expected: 5000x10) : %s' , sizzer(z3));

a3 = sigmoid(z3);
% fprintf('size a3 (expected: 5000x10) : %s' , sizzer(a3));

y_matrix = eye(num_labels)(y,:);
% fprintf('size y_matrix (expected: 5000x10) : %s' , sizzer(y_matrix));

ya3 = y_matrix .* log(a3);
ya3_2 = (1 - y_matrix) .* log(1 - a3) ;

J = (-1/m) * ( sum(sum(ya3)) + sum(sum(ya3_2)) );
% fprintf('size J (expected: 1) : %s' , sizzer(J));

% REGULARIZATION
regTerm = ( lambda/(2*m)) * (  sum( sum(dropFirstCol(Theta1).^2) ) + sum( sum(dropFirstCol(Theta2).^2) )  );
J = J + regTerm;
% fprintf('size J (expected: 1) : %s' , sizzer(J));

% (m x r)
d3 = a3 - y_matrix;
% fprintf('size d3 (expected:  5000x10) : %s' , sizzer(d3));

% (m x r) X (r x h) --> (m x h)
% (5000 x 10) X (10 x 25) --> (5000 x 25)
d2 = (d3 * dropFirstCol(Theta2)) .* sigmoidGradient(z2);
% fprintf('size d2 (expected:  5000x25) : %s' , sizzer(d2));

% (h x m) ⋅ (m x n) --> (h x n)
% (25 x 5000) X (5000 x 401) --> (25 x 401)
Delta1 = d2' * a1;
% fprintf('size my Delta1 (expected:  25x401) : %s' , sizzer(Delta1));

% (r x m) X (m x [h+1]) --> (r x [h+1])
% (10 x 5000) X (5000 x 26) --> (10 x 26)
Delta2 = d3' * a2;
% fprintf('size my Delta2 (expected:  10x26) : %s' , sizzer(Delta2));

Theta1_grad = Theta1_grad + (1/m) * Delta1;
% fprintf('size my Theta1_grad (expected:  25x401) : %s' , sizzer(Theta1_grad));

Theta2_grad = Theta2_grad + (1/m) * Delta2;
% fprintf('size my Theta2_grad (expected:  10x26) : %s' , sizzer(Theta2_grad));


% REGULARIZATION OF GRADIENT

regulator = (lambda/m);
Theta1 = zeroFirstCol(Theta1);
% fprintf('size Theta1 (expected:  25x401): %s' , sizzer(Theta1));
Theta2 = zeroFirstCol(Theta2);
% fprintf('size Theta2 (expected:  10x26): %s' , sizzer(Theta2));

Theta1_grad = Theta1_grad + (Theta1 .* regulator);
% fprintf('size Theta1_grad (expected:  25x401) : %s' , sizzer(Theta1_grad));

Theta2_grad = Theta2_grad + (Theta2 .* regulator);
% fprintf('size Theta2_grad (expected:  10x26) : %s' , sizzer(Theta2_grad));



% FUNCTIONS %%

%% set elem(1) to zero
function slyce = dropFirstCol(matrix)
slyce = matrix(:,2:end);
end

%% Prepend a column.
%% 'num' is the column fill-value
function padFirst = prependCol(num, matrix),
	[m, n] = size(matrix);
	if (num == 0),
		padFirst = [zeros(m, 1), matrix];
	else
		padFirst = [(ones(m, 1) * num), matrix];
	end
end

%% set elem(1) to zero
function slyce = zeroFirstCol(matrix)
	slyce = matrix;
	slyce(:,1) = 0;
end


function outp = sizzer(matrix),
	[m,n] = size(matrix);
		% outp = sprintf('%d x %d\n', m, n);
	if (length(matrix) < 11)
		outp = sprintf('=== (%d x %d) ====\n%s\n', m, n, mat2str(matrix));
	else
		outp = sprintf('%d x %d\n', m, n);
	end
end


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
