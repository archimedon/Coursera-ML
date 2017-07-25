function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% size(Theta1)	= (25 x 401)
% size(Theta2)	= (10 X 26 )
% size(X) 		= (5000 x 400 )

% (5000 X 26 )
hyps = calc(calc(X, Theta1), Theta2);

% [hyp_max, hyp_index] . Set 'p' to column-index of max hypThetaOfX indicating the classifier with the strongest assertion.
% column-index is equal to class
[hyp_max, p] = max(hyps, [], 2);

function hyps = calc(inputs, theta),
	[m, n] = size(inputs);
	inputs = [ones(m, 1), inputs];
	hyps = sigmoid(inputs * theta');
end

% =========================================================================


end
