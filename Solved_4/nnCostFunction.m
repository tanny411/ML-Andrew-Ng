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

X_1 = [ones(m, 1) X];
%a2=sigmoid(X_1*Theta1');
%a2_1=[ones(m,1) a2];
%a3=sigmoid(a2_1*Theta2'); %this is h--m by 10

eye_matrix = eye(num_labels);
Y = eye_matrix(y,:);

%J=sum(sum(-Y.*log(a3)-(1-Y).*log(1-a3)));

D1 = zeros(size(Theta1,1),size(Theta1,2));
D2 = zeros(size(Theta2,1),size(Theta2,2));

for i=1:m
    %forward prop
    a_1=X_1(i,:);
    a_2=[1 sigmoid(a_1*Theta1')];
    a_3=sigmoid(a_2*Theta2');
    
    %cost function
    J=J-Y(i,:)*log(a_3')-(1-Y(i,:))*log(1-a_3');
    %size(J)
    %backprop
    del_3=a_3-Y(i,:);
    del_2=(del_3*Theta2).*(a_2.*(1-a_2));
    del_2=del_2(:,2:end); %removing the del corresponding to the bias term
    
    D2=D2+del_3'*a_2;
    D1=D1+del_2'*a_1;
    
end
%regularization back-prop
D1(:,2:end) = D1(:,2:end) + lambda*Theta1(:,2:end);
D2(:,2:end) = D2(:,2:end) + lambda*Theta2(:,2:end);

%final
Theta1_grad=D1/m;
Theta2_grad=D2/m;

%regularization cost function
T1=Theta1; T1(:,1)=0;
T2=Theta2; T2(:,1)=0;
regSum=sum(sum(T1.*T1));
regSum=regSum+sum(sum(T2.*T2));
J=J + lambda/2 * regSum;

%final
J=J/m;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
