%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Linear algebra crash course
%      VIDEO: MATLAB: rank and shifting
% Instructor: sincxpress.com
%

%% a clean MATLAB workspace is a clean mental workspace

close all
clear
clc

%% ------------- %%
%   Matrix rank   %
%   -----------   %

% a few simple matrices
A = [ 1 1;
      2 2 ];

B = [ 1 1;
      1 3 ];

rankA = rank(A);
rankB = rank(B);

disp("rank(A)=" + rankA + ",  rank(B)=" + rankB)

% QUESTION: Why is matrix A reduced rank? 
%           Can you change one number in the matrix to make it full-rank?
%           Can you change one number in B to reduce its rank to 1?

%% rank of random matrices

% random matrices are basically guaranteed to be full rank, which is the 
% smallest size of the matrix:

A = randn(10,30);

rank(A)


%% --------------------- %%
%   Linear independence   %
%   -------------------   %



%%% The goal here is to build some geometric intuition for independence.
%   The green plane is the subspace spanned by the columns of matrix S.
%   The blue vector is a linear combination of those columns, and thus lies
%   in the same plane. The red vector cannot be expressed using the columns
%   of S and is therefore not in the plane.



% matrix S
S = [ 5 3; -5 1; 0 1 ];

% a vector that is not in the subspace spanned by S
v1 = [1 -2 3]';

% a vector that *is* in the subspace spanned by S
v2 = S(:,1) + 3*S(:,2);



figure(1), clf
plot3([0 S(1,1)],[0 S(2,1)],[0 S(3,1)],'k','linew',3)
hold on
plot3([0 S(1,2)],[0 S(2,2)],[0 S(3,2)],'k','linew',3)
plot3([0 v1(1)],[0 v1(2)],[0 v1(3)],'r','linew',3)
plot3([0 v2(1)],[0 v2(2)],[0 v2(3)],'b','linew',3)

% draw the plane spanned by S
h = ezmesh(@(s,t)S(1,1)*s+S(1,2)*t,@(s,t)S(2,1)*s+S(2,2)*t,@(s,t)S(3,1)*s+S(3,2)*t,[-1 1 -1 1 -1 1]);
set(h,'facecolor','g','cdata',ones(50),'linestyle','none')
xlabel('C_1'), ylabel('C_2'), zlabel('C_3')
axis square
title(' ')
grid on, rotate3d on

% QUESTION: Can you change the red vector so it's in the span of S?
% 


%% testing for independence via matrix rank

%%% One definition of rank is the largest number of columns (or rows) that
%   would form a linearly independent set.

% three column vectors
c1 = [ 1 2 3 ]';
c2 = [ 2 4 6 ]';
c3 = [ 2 4 5 ]';

% determine whether {c1,c2} is an independent set
rank([ c1 c2 ])

% repeat for {c1,c3}
rank([ c1 c3 ])

% repeat for {c1,c2,c3}
M = [ c1 c2 c3 ];
rank(M)


% QUESTION: Does the rank change for transposed matrices? Try it!
% 


%% --------------------- %%
%   "Shifting" a matrix   %
%   -------------------   %


%% shifting a matrix

% "shifting" a matrix
l = .3; % lambda
N =  5; % size of square matrix
D =  randn(N);   % can only shift a square matrix

% the shifting
Ds = D + l*eye(N);


%% "shifting" a matrix to full-rank

% Create a reduced-rank matrix. The mechanics of why this works is from the
% theory of matrix rank. It wasn't covered in class.

r = 5; % desired rank of the matrix
m = 12; % size of square matrix. make this higher than r

redRankMat = randn(m,r)*randn(r,m);


% now shift the matrix by lambda
lambda = .01;
shiftedMatrix = redRankMat + lambda*eye(m);


% display their ranks
disp([ 'The rank of the original matrix is ' num2str(rank(redRankMat)) ])
disp([ 'The rank of the shifted  matrix is ' num2str(rank(shiftedMatrix)) ])


% QUESTION: What is the smallest you can make lambda and still get a
%           full-rank matrix? (Hint: scientific notation is 1e-5 for .00001)
%           What does this tell you about making full-rank matrices?


%% done.
