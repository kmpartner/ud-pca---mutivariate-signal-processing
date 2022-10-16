%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Linear algebra crash course
%      VIDEO: MATLAB: eig and svd
% Instructor: sincxpress.com
%

%% a clean MATLAB workspace is a clean mental workspace

close all
clear
clc

%% ---------------------- %%
%    Eigendecomposition    %
%   --------------------   %



%%% This cell shows a geometric picture of eigenvectors in 2D.
%   Notice that the two red lines (v and Av) are collinear, 
%   but not the two black lines (w and Aw). v is an eigenvector, w is not.


% matrix
A = [1 5; 2 4];

% extract the eigenvalues
eigvals = eig(A);


% specify two vectors
v = [ 1 1 ]';   % is an eigenvector!
w = randn(2,1); % unlikely to be an eigenvector
w = w/norm(w);  % unit length for convenience

% compute Av
Av = A*v;
Aw = A*w;


% plot the vectors and Av
figure(1), clf
plot([0 v(1)],[0 v(2)],'r','linew',4)
hold on
plot([0 Av(1)],[0 Av(2)],'r--','linew',2)
plot([0 w(1)],[0 w(2)],'k','linew',4)
plot([0 Aw(1)],[0 Aw(2)],'k--','linew',2)
legend({'v (evec.)';'A\timesv';'w (not evec.)';'A\timesw'})

lim = max([Av(:); Aw(:)])*1.2;
axis([-1 1 -1 1]*lim)
grid on, axis square
plot(get(gca,'xlim'),[0 0],'k')
plot([0 0],get(gca,'xlim'),'k')


%%% QUESTION: Is there something special about vector v? To find out,
%             change one value of matrix A.



%% -------------------------------------- %%
%    Eigenvectors of symmetric matrices    %
%   ------------------------------------   %


%%% Symmetric matrices have many special properties. 
%   One of them is that they have orthogonal eigenvectors, 
%   which you will see in this cell.


% create a random matrix
X = randn(14);

% make it symmetric
S = X'*X;

% diagonalize it
[evecs,evals] = eig(S);


% magnitudes of each vector
sqrt( sum(evecs.^2,1) )

%%

% and make plots
figure(2), clf
subplot(131), imagesc(S)
axis square, axis off
title('S')

subplot(132), imagesc(evecs)
axis square, axis off
title('Eigenvectors')

subplot(133), imagesc(evecs*evecs')
axis square, axis off
title('VV^T')

%%% QUESTION: If VtV = I, then Vt = V^-1. How can you verify this in MATLAB?
% 
% Answer: you can inspect evecs' - inv(evecs). Note that due to computer
% rounding errors, 1e-15 (10^-15) can be considered zero.

%% side note about the eig() function

% with one output, eig returns a vector of eigenvalues
lambdas = eig(S);

% with two outputs, you get the matrices
[V,D] = eig(S);

% note the relationship between lambdas and D:
size(lambdas)
size(D)

[ lambdas diag(D) ]

%% some fun with random matrices

% Random matrices tend to have complex eigenvalues
% Those eigenvalues form a circle in the complex plane.

nIter = 150;
matsize = 40;
evals = zeros(nIter,matsize);

for i=1:nIter
    X = randn(matsize);
    evals(i,:) = eig(X);
end


figure(3), clf
plot(real(evals(:)),imag(evals(:)),'ro','markerfacecolor','k')
xlabel('Real part'), ylabel('Imaginary part')
axis square
lim = max(abs(evals(:)));
axis([-1 1 -1 1]*lim*1.1)


%%% Note: This is just a fun FYI; no practical applications for this course.
% 

%% done with eigendecomposition code





%% a clean MATLAB workspace is a clean mental workspace

close all
clear
clc

%% -------------------------------------- %%
%    Singular value decomposition (SVD)    %
%   ------------------------------------   %

% Here we will show that eigendecomposition and SVD are the same thing for
% symmetric matrices (they are *not* the same for other kinds of matrices;
% this is an example of a special property of symmetric matrices).

% create a symmetric matrix
X = randn(5);
S = X'*X;

% take its eigendecomposition
[W_eig,L_eig] = eig(S);

% take its SVD
[U_svd,S_svd,V_svd] = svd(S);


% If they are identical, their difference should be zero:
W_eig - U_svd

% deffo not zero...

%% eigenvalues and singular values

[ diag(L_eig) diag(S_svd) ]
% ah, they need to be sorted ;)

[l_eig,sidx] = sort(diag(L_eig),'desc');

% now try again:
[ l_eig diag(S_svd) ]

% now try the vectors
W_eig(:,sidx) - U_svd

% I guess some columns are zero and others not? The issue now is the sign
% uncertainty of eigenvectors:
(-W_eig(:,sidx)) - U_svd



%% ------------------- %%
%    SVD of Einstein    %
%   -----------------   %


%%% The goal here is to visualize the SVD matrices.


% import the picture
ein = imread('einstein.jpg');

% view it
figure(1), clf
imagesc(ein), axis image, axis off

% pictures are 3D (third dimension is color), but we want a "flat" matrix.
einflat = mean(ein,3);


% why doesn't eigendecomposition work?
% [V,D] = eig(einflat);



% SVD of image
[U,S,V] = svd(einflat);

% image the three matrices
figure(2), clf
subplot(231), imagesc(U), axis image
title('U (left singular vectors)')

subplot(232), imagesc(S), axis image
set(gca,'clim',[0 200])
title('S (singular values)')

subplot(233), imagesc(V'), axis image
title('V^T (right singular vectors)')


% show the singular values, "raw" and percent change
singvals    = diag(S);
singvalspct = 100*singvals./sum(singvals);

subplot(223)
plot(singvals,'s-','markerfacecolor','w')
set(gca,'xlim',[0 50]), axis square
title('First 50 singular values')
xlabel('Component number'), ylabel('\sigma')


subplot(224)
plot(singvalspct,'s-','markerfacecolor','w')
set(gca,'xlim',[0 50]), axis square
title('First 50 singular values')
xlabel('Component number'), ylabel('% variance explained')

%% low-rank approximations


%%% The goal here is to recreate Einstein using a small number of the most
%   important feature dimensions.

figure(3), clf


for i=1:16
    
    % low-rank approximation
    lowapp = U(:,1:i) * S(1:i,1:i) * V(:,1:i)';
    
    % compute its rank
    matrix_rank = rank(lowapp);
    
    % visualize!
    subplot(4,4,i)
    imagesc(lowapp), axis off, axis image
    title({ "r=" + matrix_rank ; sprintf('%.0f%% var.',sum(singvalspct(1:i))) },'fontsize',8)
    
end
colormap gray


% QUESTION: How many components / what percent variance do you need to
%           get a "good" (subjective) reconstruction of Einstein? 
%           Does this give you any hope for post-mortem resurrection??


%% done.
