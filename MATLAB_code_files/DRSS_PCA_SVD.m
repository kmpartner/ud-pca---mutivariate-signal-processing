%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Creating and interpreting covariance matrices
%      VIDEO: MATLAB: PCA via SVD of covariance
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%% simulate data with covariance structure

% simulation parameters
N = 10000; % time points
M =    20; % channels

% time vector (radian units)
t = linspace(0,6*pi,N);

% relationship across channels (imposing covariance)
chanrel = sin(linspace(0,2*pi,M))';

% generate data
data = bsxfun(@times,repmat( sin(t),M,1 ),chanrel) + randn(M,N);

%% PCA via eigendecomposition of covariance matrix

% first mean-center (mean over time!)
dataM = data - mean(data,2);

% all pairwise dot products as the matrix times its transpose
covmat = dataM*dataM'/(N-1);

% sort
[Veig,Deig] = eig( covmat );
[Deig,sidx] = sort( diag(Deig),'descend' );
Veig = Veig(:,sidx);

% convert eigenvalues to %
Deig = 100*Deig/sum(Deig);

% time series of top component
eig_ts = Veig(:,1)'*data;

%% PCA via SVD of data matrix

% the SVD
[U,S,V] = svd( dataM,'econ' );

% convert singular values to %
S = S.^2; % makes it comparable to eigenvalues
s = 100*diag(S)/sum(diag(S));

%% show the results

figure(1), clf

% plot eigenvalue/singular-value spectrum
subplot(221), hold on
plot(Deig,'bs-','markerfacecolor','w','markersize',10,'linew',2)
plot(s,'ro-','markerfacecolor','w','markersize',5)
xlabel('Component number'), ylabel('\lambda or \sigma')
legend({'eig';'svd'})
title('Eigenspectrum')


% show eigenvector/singular value
subplot(222), hold on
plot(Veig(:,1),'bs-','markerfacecolor','w','markersize',10,'linew',2)
plot(-U(:,1),'ro-','markerfacecolor','w','markersize',5)
xlabel('Vector component'), ylabel('Weight')
title('Component weights')


% time series
subplot(212)
timevec = (0:N-1)/N;
plot(timevec,eig_ts, timevec,-V(:,1)*sqrt(S(1)),'.')
xlabel('Time (norm.)')
title('Component time series')
legend({'eig';'svd'})
zoom on

%%
