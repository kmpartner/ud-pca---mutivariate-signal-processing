%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Linear algebra crash course
%      VIDEO: MATLAB: vectors and matrices
% Instructor: sincxpress.com
%

%% a clean MATLAB workspace is a clean mental workspace

close all
clear
clc

%% --------------------- %%
%   Visualizing vectors   %
%   -------------------   %


% create a 2-dimensional vector
v2 = [ 3 -2 ];

% create a 3-dimensional vector
v3 = [ 4 -3 1 ];

% plot them
figure(1), clf
subplot(121)

% the main plotting line:
plot([0 v2(1)],[0 v2(2)],'linew',2)

% this code just makes the plot look nicer
axis square
axis([ -4 4 -4 4 ])
hold on
plot(get(gca,'xlim'),[0 0],'k--')
plot([0 0],get(gca,'ylim'),'k--')
xlabel('X_1 dimension')
ylabel('X_2 dimension')
title('Vector in R^2')


% plot the 3D vector
subplot(122)

% the main plotting code:
plot3([0 v3(1)],[0 v3(2)], [0 v3(3)],'linew',2)

% make the plot look nicer
axis square
axis([ -4 4 -4 4 -4 4 ])
hold on, grid on
plot3(get(gca,'xlim'),[0 0],[0 0],'k--')
plot3([0 0],get(gca,'ylim'),[0 0],'k--')
plot3([0 0],[0 0],get(gca,'zlim'),'k--')
xlabel('X_1 dimension')
ylabel('X_2 dimension')
zlabel('X_3 dimension')
title('Vector in R^3')

% might be easier to see when rotated
rotate3d on 

%% vector-scalar multiplication


% 2D vectors
v = [ 3 -1 ];

% scalar
s = 2;

% make the new vector
sv = s*v;


% plot them
figure(2), clf

% the main plotting lines:
plot([0 v(1)],[0 v(2)],'b','linew',4)
hold on
plot([0 sv(1)],[0 sv(2)],'r--','linew',2)
legend({'v';'sv'})

% this code just makes the plot look nicer
axis square
axis([ -1 1 -1 1 ]*max([norm(sv) norm(v)]))
hold on
plot(get(gca,'xlim'),[0 0],'k--')
plot([0 0],get(gca,'ylim'),'k--')
xlabel('X_1 dimension')
ylabel('X_2 dimension')


% QUESTION: What happens when the scalar is >1, <1, <0, and =0?
% 

%% transpose operation

v3

% transpose
v3'

% double-transpose to confirm it equals the original vector


% observe the sizes
size( v3  )
size( v3' )


%% --------------------------- %%
%   OMG it's the dot product!   %
%   ------------------------    %

% the two vectors
v1 = [ 1 2 3 4 5 6 ];
v2 = [ 0 8 -4 -3 6 5 ];

% the manual method: element-wise multiply and sum
dp = sum( v1.*v2 );

% the dot function
dp = dot( v1,v2 );


% QUESTION: Is the dot product affected by where you scalar-multiply?
%           To find out, create a scalar s and use it to multiply:
%           1) v1
%           2) v2
%           3) dot(v1,v2)
%           Do the results differ?




%% ------------------------------------ %%
%   Creating and working with matrices   %
%   ---------------------------------    %


% matrix sizes
m = 14;
n = 8;

% random matrix
A = randn(m,n);
size(A)


% MxM identity matrix
I = eye(m)

% diagonal
d = [ 1 2 4 9 15 ];
D = diag(d);
dd = diag(D);

% symmetric
S = A'*A; % you'll learn later why this is symmetric
size(A) % size of A
size(S) % size of S


% if the matrix is symmetric, then it should equal its transpose:
S' - S

% QUESTION: We haven't yet discussed diagonal or symmetric matrices. 
%           But based on this code, what do you think are the definitions
%           of diagonal and symmetric matrices?
%           Hint: for symmetric matrices, think about the sizes of matrix
%           multiplications.
% 


%% matrix arithmetic (addition and scalar multiplication)

% create random matrices
A = randn(5,4);
B = randn(5,3);
C = randn(5,4);

% try to add them
A+B
A+C


% scalar multiplication
M = round( randn(2,3)*10 );
s = 2;

M

% compute and confirm that sM==Ms
s*M
M*s


%% ------------------------ %%
%   Matrix multiplications   %
%   ----------------------   %


% matrix sizes
m = 4;
n = 3;
k = 6;

% make some matrices
A = randn(m,n);
B = randn(n,k);
C = randn(m,k);

% test which multiplications are valid.
% Think of your answer first, then test!
A*B;
A*A;
A'*C;
B*B';
B'*B;
B*C;
C*B;
C'*B;
C*B';


%% LIVE EVIL!

%%% Here you can see the law of reversing the order of matrices when
%   operating on them.

% create some random matrices
A = randn(3);
B = randn(3);
C = randn(3);

% LIVE
(A*B*C)'

% EVIL
C'*B'*A'



%%% this rule also works for other operations such as the inverse:
inv(A*B*C)
inv(C)*inv(B)*inv(A)

%% done.
