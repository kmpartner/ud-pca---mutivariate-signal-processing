%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Linear algebra crash course
%      VIDEO: MATLAB: inverse and AtA
% Instructor: sincxpress.com
%

%% a clean MATLAB workspace is a clean mental workspace

close all
clear
clc

%% ---------------- %%
%   Matrix inverse   %
%   --------------   %


% size of square matrix
m = 3;

% generate random matrix
A = randn(m);

% compute its inverse
Ainv = inv(A);

% check whether the matrix times its inverse is the identity matrix
id = Ainv*A

% QUESTION: Does A*Ainv also produce the identity matrix?


% show in an image
figure(1), clf
subplot(131), imagesc(A)
title('Matrix A'), axis square, axis off

subplot(132), imagesc(Ainv)
title('Matrix A^{-1}'), axis square, axis off

subplot(133), imagesc(id)
title('AA^{-1}'), axis square, axis off



% QUESTION: What happens when you run the code above using the following
%           reduced-rank matrix?
A = randn(4,2) * randn(2,4);


%% The left inverse

% sizes
m = 6;
n = 3;

% create matrices
A = randn(m,n);
AtA = A'*A;

% inspect ranks
disp([ 'Rank of A^TA: ' num2str(rank(AtA)) ])

% left inverse
Aleft = inv(AtA)*A';

% now test that the left-inverse times A is identity!
I_left  = Aleft*A;

% and then test using the inverse function
AtA_inv = inv(AtA);
I_AtA   = AtA_inv*AtA;

% QUESTION: What happens when you right-multiply the left-inverse?




% show images
figure(2), clf
subplot(221), imagesc(A), axis image, axis off
title([ 'A, r=' num2str(rank(A)) ])

subplot(222), imagesc(AtA), axis image, axis off
title([ 'A^TA, r=' num2str(rank(AtA)) ])

subplot(223), imagesc(Aleft), axis image, axis off
title('Left inverse: (A^TA)^{-1}A^T')

subplot(224), imagesc(I_left), axis image, axis off
title('[ (A^TA)^{-1}A^T ]  A')

%% right inverse

% Repeat the above code for the right-inverse!






%% -------------------------- %%
%   The wonderful matrix AtA   %
%   ------------------------   %


% show that AtA is square

% sizes
m = 6;
n = 3;

% compute it both ways
AtA = A'*A;
AAt = A*A';


disp(size(AtA))
disp(size(AAt))


%% show that it's symmetric

% show the matrices
figure(3), clf

subplot(231)
imagesc(A)
axis square
title('A')

subplot(232)
imagesc(AtA)
axis square
title('A^TA')

subplot(233)
imagesc(AtA - AtA')
axis square
title('A^TA  -  (A^TA)^T')



subplot(234)
% repeat the previous three plots but for AAt and in the bottom row
imagesc(A)
axis square
title('A')

subplot(235)
imagesc(AAt)
axis square
title('AA^T')

subplot(236)
imagesc(AAt - AAt')
axis square
title('AA^T  -  (AA^T)^T')

%% done.
