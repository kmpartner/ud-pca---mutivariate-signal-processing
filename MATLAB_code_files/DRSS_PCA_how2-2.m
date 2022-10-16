%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Dimension reduction with principal components analysis
%      VIDEO: How to perform a principal components analysis
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc


%% simulate data with covariance structure

% simulation parameters
N = 1000;     % time points
M =   20;     % channels
nTrials = 50; % number of trials

% time vector (radian units)
t = linspace(0,6*pi,N);


% relationship across channels (imposing covariance)
chanrel = sin(linspace(0,2*pi,M))';

data = bsxfun(@times,repmat( sin(t),M,1 ),chanrel) + randn(M,N);



% step 1: mean-center and compute covariance matrix
data = bsxfun(@minus,data,mean(data,2));
covmat = data*data'/(N-1);


% step 2: eigendecomposition
[evecs,evals] = eig( covmat );


% step 3: sort vectors by values
[evals,soidx] = sort( diag(evals),'descend' );
evecs = evecs(:,soidx);


% step 4: compute component time series
r = 2; % two components
comp_time_series = evecs(:,1:r)'*data;


% step 5: convert eigenvalues to percent change
evals = 100*evals./sum(evals);


% step 6: visualize and interpret the results
figure(1), clf

% eigenvalues
subplot(231)
plot(evals,'s-','linew',2,'markerfacecolor','w')
axis square
xlabel('Component number'), ylabel('\lambda')

% eigenvectors
subplot(232)
plot(evecs(:,1:2),'s-','linew',2,'markerfacecolor','w')
axis square
xlabel('Channel'), ylabel('PC weight')
legend({'PC1';'PC2'})

% original channel modulator
subplot(233)
plot(chanrel,'s-','linew',2,'markerfacecolor','w')
axis square
xlabel('Channel'), ylabel('True channel weights')

% component time series
subplot(212)
plot(1:N,comp_time_series)
xlabel('Time (a.u.)'), ylabel('Activity')
legend({'PC1';'PC2'})

%%






data2 = cell(1,2);

% relationship across channels (imposing covariance)
chanrel = sin(linspace(0,2*pi,M))';


% loop over "trials" and generate data
for triali=1:nTrials
    
    % simulation 1
    data2{1}(:,:,triali) = bsxfun(@times,repmat( sin(t),M,1 ),chanrel) + randn(M,N)/1;
    
    % simulation 2
    data2{2}(:,:,triali) = bsxfun(@times,repmat( sin(t+rand*2*pi),M,1 ),chanrel) + randn(M,N)/1;
end

%% visual inspection of the time series data

figure(2), clf

for i=1:2
    subplot(1,2,i)
    
    % pick a random trial
    trial2plot = ceil(rand*nTrials);
    
    % show the data
    plot(t,bsxfun(@plus, squeeze(data2{i}(:,:,trial2plot)) ,(1:M)'*3 ))
    
    % code to show trials from one channel
    %plot(t,bsxfun(@plus, squeeze(data{i}(5,:,1:M)) ,(1:M)*3 ))
    
    % make the plot look a bit nicer
    set(gca,'ytick',[]), axis tight
    xlabel('Time'), ylabel('Channels')
    title([ 'Dataset ' num2str(i) ', trial' num2str(trial2plot) ])
    
    % code if showing trials instead of channels
    %ylabel('Trials')
    %title([ 'Dataset ' num2str(i) ', channel 5' ])
end

figure(3), clf
subplot(111)
imagesc(covmat), axis square
set(gca,'clim',[-1 1],'xtick',1:20,'ytick',1:20)
xlabel('Channels'), ylabel('Channels')
title([ 'covmat' ])
