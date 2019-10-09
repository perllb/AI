function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Cvec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
SigVec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
scores = zeros(length(Cvec)*length(SigVec),3)
idx = 0

for i = 1:length(Cvec)
    
    for j = 1:length(SigVec)
        
        idx = idx + 1;
        
        CurrModel = svmTrain(X, y, Cvec(i), @(x1, x2) gaussianKernel(x1, x2, SigVec(j))); 
        
        predictions = svmPredict(CurrModel,Xval);
        
        error = mean(double(predictions ~= yval));
        
        scores(idx,1) = Cvec(i);
        scores(idx,2) = SigVec(j);
        scores(idx,3) = error;
        
    end
    
end
        
minim = min(scores(:,3));    
row = find(scores(:,3)==minim);

C = scores(row,1);
sigma = scores(row,2);






% =========================================================================

end
