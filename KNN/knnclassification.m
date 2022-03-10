function result = knnclassification(testsamplesX,samplesX, samplesY, Knn,type)

% Classify using the Nearest neighbor algorithm
% Inputs:
% 	samplesX	   - Train samples
%   testsamplesX   - Test  samples
%	Knn		       - Number of nearest neighbors 
%
% Outputs
%	result	- Predicted targets
if nargin < 5
    type = '2norm';
end

L			= length(samplesY);%erroe?,length(samplesY),no ,for X,Y's 1dim is equal
Uc          = unique(samplesY);%类别数目

if (L < Knn),
   error('You specified more neighbors than there are points.')
end

N                   = size(testsamplesX, 1);
result              = zeros(N,1); 
switch type
case '2norm'
    for i = 1:N,
        dist            = sum((samplesX - ones(L,1)*testsamplesX(i,:)).^2,2);
        [m, indices]    = sort(dist);%SORT(X) sorts the elements of X in ascending order   
        n               = hist(samplesY(indices(1:Knn)), Uc);%ji suan jie jin na yi lei
        [m, best]       = max(n);
        result(i)        = Uc(best);
    end
case '1norm'
    for i = 1:N,
        dist            = sum(abs(samplesX - ones(L,1)*testsamplesX(i,:)),2);
        [m, indices]    = sort(dist);   
        n               = hist(samplesY(indices(1:Knn)), Uc);
        [m, best]       = max(n);
        result(i)        = Uc(best);
    end
case 'match'
    for i = 1:N,
        dist            = sum(samplesX == ones(L,1)*testsamplesX(i,:),2);
        [m, indices]    = sort(dist);   
        n               = hist(samplesY(indices(1:Knn)), Uc);
        [m, best]       = max(n);
        result(i)        = Uc(best);
    end
otherwise
    error('Unknown measure function');
end
