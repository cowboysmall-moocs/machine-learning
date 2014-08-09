function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


reg_vector = ones(1, size(theta)(1));
reg_vector(1, 1) = 0;

J = ((-y' * log(sigmoid(X * theta)) - ((1 - y)' * log(1 - sigmoid(X * theta)))) / m) + (((lambda * reg_vector * theta.^2)) / (2 * m));


reg_matrix = eye(size(theta)(1));
reg_matrix(1, 1) = 0;

grad = (((sigmoid(X * theta) - y)' * X) + (lambda * theta' * reg_matrix)) / m;


% =============================================================

end
