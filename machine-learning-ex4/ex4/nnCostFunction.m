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
% printf('info [a1] :-  (expected: 5000x401) : %s' , varDump(a1, 1));

% (m x n) X (n x h) --> (m x h)
z2 = a1 * Theta1';
% printf('info [z2] :-  (expected: 5000x25) : %s' , varDump(z2, 3));

a2 = sigmoid(z2);
% printf('info [a2] :-  (expected: 5000x25) : %s' , varDump(a2, 3));

a2 = prependCol(1, a2);
% printf('Add bias.. size a2 (expected: 5000x26) : %s' , varDump(a2));

z3 = a2 * Theta2';
% printf('info [z3] :-  (expected: 5000x10) : %s' , varDump(z3, 3));

a3 = sigmoid(z3);
% printf('info [a3] :-  (expected: 5000x10) : %s' , varDump(a3, 3));

y_matrix = eye(num_labels)(y,:);
% printf('info [y_matrix] :-  (expected: 5000x10) : %s' , varDump(y_matrix));

% Invert for convenience
m_inv = (1/m);


ya3_pt1 = y_matrix .* log(a3);
ya3_pt2 = (1 - y_matrix) .* log(1 - a3) ;

J = (- m_inv) * ( sum(sum(ya3_pt1)) + sum(sum(ya3_pt2)) );
% printf('info [J] :-  (expected: 1) : %s' , varDump(J));

% REGULARIZATION OF COST

%%%%%
% First: Don't regularize the bias term (created initiially at X[0]*theta[0], or later at a[0]*theta2[0])
regTerm = ( lambda/(2*m)) * (  sum( sum(dropFirstCol(Theta1).^2) ) + sum( sum(dropFirstCol(Theta2).^2) )  );
J = J + regTerm;
% printf('info [J] :-  (expected: 1) : %s' , varDump(J));

% (m x r)
d3 = a3 - y_matrix;
% printf('info [d3] :-  (expected:  5000x10) : %s' , varDump(d3));

% (m x r) X (r x h) --> (m x h)
% (5000 x 10) X (10 x 25) --> (5000 x 25)
d2 = (d3 * dropFirstCol(Theta2)) .* sigmoidGradient(z2);
% printf('info [d2] :-  (expected:  5000x25) : %s' , varDump(d2));
% EQUIVALENT :
% d2 = dropFirstCol(  (d3 * Theta2) .* a2 .* (1 - a2)  );

% Iterative form would be:
%	Delta[l] = Delta[l] + d[l + 1] * transpose(a[l])
%
% (h x m) ⋅ (m x n) --> (h x n)
% (25 x 5000) X (5000 x 401) --> (25 x 401)
Delta1 = derivCostJ_l1 = d2' * a1;				% A matrix of GRADIENT SUMS for the nodes in layer-1
% printf('info [my Delta1] :-  (expected:  25x401) : %s' , varDump(Delta1));

% (r x m) X (m x [h+1]) --> (r x [h+1])
% (10 x 5000) X (5000 x 26) --> (10 x 26)
Delta2 = derivCostJ_l2 = d3' * a2;				% A matrix of GRADIENT SUMS for the nodes in layer-2
% printf('info [my Delta2] :-  (expected:  10x26) : %s' , varDump(Delta2));

Theta1_grad = Theta1_grad + m_inv * Delta1;
% printf('info [my Theta1_grad] :-  (expected:  25x401) : %s' , varDump(Theta1_grad));

Theta2_grad = Theta2_grad + m_inv * Delta2;
% printf('info [my Theta2_grad] :-  (expected:  10x26) : %s' , varDump(Theta2_grad));


% REGULARIZATION OF GRADIENT

regulator = (lambda/m);
Theta1 = zeroFirstCol(Theta1);
% printf('info [Theta1] :-  (expected:  25x401): %s' , varDump(Theta1));
Theta2 = zeroFirstCol(Theta2);
% printf('info [Theta2] :-  (expected:  10x26): %s' , varDump(Theta2));

Theta1_grad = Theta1_grad + (Theta1 .* regulator);
% printf('info [Theta1_grad] :-  (expected:  25x401) : %s' , varDump(Theta1_grad));

Theta2_grad = Theta2_grad + (Theta2 .* regulator);
% printf('info [Theta2_grad] :-  (expected:  10x26) : %s' , varDump(Theta2_grad));



% FUNCTIONS %%

function slyce = dropFirstCol(matrix)
% set elem(1) to zero
	slyce = matrix(:,2:end);
end

function padFirst = prependCol(num, matrix),
% Prepend a column.
% 'num' is the column fill-value
	[m, n] = size(matrix);
	if (num == 0),
		padFirst = [zeros(m, 1), matrix];
	else
		padFirst = [(ones(m, 1) * num), matrix];
	end
end

function slyce = zeroFirstCol(matrix)
% set elem(1) to zero
	slyce = matrix;
	slyce(:,1) = 0;
end

function outp = varDump(matrix, peek),
% Show size of variable or values if 10x10 or less.
%
% Params:
%  	matrix 	- the variable
% 	peek 	- number of rows to show
%
	[m,n] = size(matrix);
	if ( exist("peek", "var") && ~isempty(peek) ),
		cstring = '%.2f';
		for ( i = 1:n-1) cstring = [cstring ' %.2f']; end
		% strcmp(typeinfo(g),'matrix')	
		therows = sprintf([cstring '\n'], matrix(1:peek,:));
		outp = sprintf('=== (%d x %d) === The first %d rows:\n%s\n', m, n, peek, therows);
	else
		% outp = sprintf('%d x %d\n', m, n);
		if (length(matrix) < 11)
			outp = sprintf('=== (%d x %d) ===\n%s\n', m, n, mat2str(matrix));
		else
			outp = sprintf('%d x %d\n', m, n);
		end
	end
end


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
