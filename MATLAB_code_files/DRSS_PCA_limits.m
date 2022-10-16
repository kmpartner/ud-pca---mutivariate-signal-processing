%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Dimension reduction with principal components analysis
%      VIDEO: Limitations of principal components analysis
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%% generate data

x = [ randn(1000,1) .05*randn(1000,1) ];

% rotation matrix
th = -pi/6;
R1 = [ cos(th) -sin(th); sin(th) cos(th) ];
th = -pi/3;
R2 = [ cos(th) -sin(th); sin(th) cos(th) ];

% rotate data
y = [ x*R1 ; x*R2 ];


figure(1), clf
subplot(121)
plot(y(:,1),y(:,2),'bo','markerfacecolor','y')
axis([-1 1 -1 1]*max(abs(y(:))))
xlabel('X axis'), ylabel('Y axis')
axis square
title('Data space')


%% run PCA

% mean-center
y = y - mean(y,1);

% eigendecomposition of covariance matrix
[evecsY,evalsY] = eig( (y'*y) / length(y) );

% plot eigenvectors
hold on
plot([0 evecsY(1,1)],[0 evecsY(2,1)],'r','linew',5)
plot([0 evecsY(1,2)],[0 evecsY(2,2)],'k','linew',5)

%% in PC space

comps = evecsY*y';

subplot(122)
plot(comps(2,:),comps(1,:),'bo','markerfacecolor','y')
axis([-1 1 -1 1]*max(abs(comps(:))))
xlabel('PC_1 axis'), ylabel('PC_2 axis')
axis square
title('PC space')

%%
