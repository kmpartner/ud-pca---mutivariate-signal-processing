%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Creating and interpreting covariance matrices
%      VIDEO: MATLAB: Visualizing the quadratic form
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%%

% create a little matrix
C = [1 2; 2 9];

% other interesting matrices to try:
%C = [2 1; 4 2];
%C = [2 1; 1 -1];
%C = [1 0; 0 1];

% weights along each dimension
wi = -2:.1:2;

quadform = zeros(length(wi));
for i=1:length(wi)
    for j=1:length(wi)
        
        % define the weights for this iteration
        w = [wi(i) wi(j)]';
        
        % normalization factor
        normfact = w'*w;
        
        % compute the quadratic form of the matrix
        quadform(i,j) = w'*C*w / normfact;
    end
end

figure(1), clf
% make a rotatable image
surf(wi,wi,quadform'), shading interp
title('Visual representation of quadratic form of matrix C')
xlabel('W1'), ylabel('W2'), zlabel('energy')
rotate3d on


% an aside: the solution vectors to the normalized quadratic form
%           are the eigenvectors of the matrix.
 [evecs,evals] = eig(C);
 hold on
 plot3([0 evecs(1,1)],[0 evecs(2,1)],[1 1]*max(quadform(:)),'w','linew',3)
 plot3([0 evecs(1,2)],[0 evecs(2,2)],[1 1]*max(quadform(:)),'m','linew',3)

%%
