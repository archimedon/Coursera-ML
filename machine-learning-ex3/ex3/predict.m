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

[m, n] = size(X);

% size(Theta1) = (25 x n + 1)
% size(Theta2) = (10 X 26 )


X = [ones(m, 1), X];

%  X: (5000 x n + 1); Theta1: (25 x n + 1)
hypThetaOfX = sigmoid(X * Theta1' );		% size(hypThetaOfX) = (5000 X 25 )


[m, n] = size(hypThetaOfX);

newInputs = [ones(m , 1), hypThetaOfX ];
% size(newInputs)
% (5000 X 26 )

predictions = sigmoid(newInputs * Theta2' );

% [hyp_max, hyp_index] . Set 'p' to column-index of max hypThetaOfX indicating the classifier with the strongest assertion.
% column-index is equal to class
[hyp_max, p] = max(predictions, [], 2);



% =========================================================================


end
