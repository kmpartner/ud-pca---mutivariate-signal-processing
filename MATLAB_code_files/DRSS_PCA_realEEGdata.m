%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Dimension reduction with PCA
%      VIDEO: MATLAB: PCA of real EEG data
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%% load data

load sampleEEGdata.mat
EEG.data = double( EEG.data );

%% compute PCA on ERP

% comptue the ERP
erp = mean(EEG.data,3);

% mean-centre and get covar
data = erp - mean(erp,2);
covd = data*data'/(EEG.pnts-1);

% eig
[evecs,evals] = eig( covd );

% sort according to eigenvalues
[evals,sidx] = sort(diag(evals),'descend');
evecs = evecs(:,sidx);
evals = 100*evals/sum(evals);

% principal component time series
pc_timeseries = evecs(:,1)'*erp;



%% plot results and compare with electrode

figure(1), clf

% eigenspectrum
subplot(221)
plot(evals(1:20),'ko-','markerfacecolor','w','linew',2)
title('Eigenspectrum'), axis square
ylabel('Percent variance'), xlabel('Component number')

% topographical map of first eigenvector
subplot(222)
##topoplotIndie(evecs(:,1),EEG.chanlocs,'numcontour',0,'shading','interp','electrodes','numbers');
topoplotIndieOctave(evecs(:,1),EEG.chanlocs,'numcontour',0,'shading','interp','electrodes','numbers');
title('PC topomap')

% plot time series
subplot(212)
plot(EEG.times,pc_timeseries, ...
     EEG.times,mean(erp([19 32 20],:),1)*5 ,'linew',2 )
##plot(EEG.times,pc_timeseries, ...
##     EEG.times,mean(erp([42 43],:),1)*5 ,'linew',2 )
legend({'PC';'Chans'})
xlabel('Time (s)')

%%

figure(2), clf
imagesc(covd)
title('covd')
colorbar
