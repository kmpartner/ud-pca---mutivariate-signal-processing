%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Dimension reduction with PCA
%      VIDEO: MATLAB: PCA on multi-trial data
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%% load data

load sampleEEGdata.mat
EEG.data = double( EEG.data );

%% erp ("phase-locked") covariance matrix

% time window to use
tidx = dsearchn(EEG.times',[0 800]');

% phase-locked covariance
erp  = mean(EEG.data(:,tidx(1):tidx(2),:),3);
data = erp - mean(erp,2);
cov_phaselocked = data*data'/(EEG.pnts-1);


figure(1), clf
subplot(121)
imagesc(cov_phaselocked)
axis square
set(gca,'clim',[-1 1]*2)
xlabel('Channels'), ylabel('Channels')
colorbar
title('Covariance of average')

%% single-trial ("total") covariance matrix

cov_total = zeros(EEG.nbchan);

for triali=1:EEG.trials
    
    % extract data and compute mini-cov
    minicov = cov( EEG.data(:,tidx(1):tidx(2),triali)' );
    
    % sum into covariance
    cov_total = cov_total + minicov;
end

% scale by N
cov_total = cov_total/triali;


% visualize
subplot(122)
imagesc(cov_total)
axis square
set(gca,'clim',[-1 1]*100)
xlabel('Channels'), ylabel('Channels')
colorbar
title('Average of covariances')

%% compute and visualize phase-locked PCA

% PCA of trial-average
[V_PL,L_PL] = eig( cov_phaselocked );
[L_PL,sidx] = sort(diag(L_PL),'descend');
V_PL = V_PL(:,sidx);
L_PL = 100*L_PL/sum(L_PL);

% principal components time series
% NOTE: applying filter to the entire time series!
compts_PL = V_PL(:,1)' * mean(EEG.data,3);



% visualize
figure(2), clf
subplot(321)
plot(L_PL,'ks-','markerfacecolor','w','markersize',8)
set(gca,'xlim',[.5 20])
xlabel('Component number')
ylabel('Pct variance')
title('Scree plot')

subplot(322)
plot(cumsum(L_PL),'ks-','markerfacecolor','w','markersize',8)
set(gca,'xlim',[.5 20])
xlabel('Component number')
ylabel('Cumulative sum variance')

% topoplot
subplot(323)
##topoplotIndie(V_PL(:,1),EEG.chanlocs,'numcontour',0,'shading','interp','electrodes','off');
topoplotIndieOctave(V_PL(:,1),EEG.chanlocs,'numcontour',0,'shading','interp','electrodes','off');
title('Phase-locked c_1')
colormap jet

% time course
subplot(313)
plot(EEG.times,compts_PL,'linew',2)
set(gca,'xlim',[-200 1300])
xlabel('Time (ms)')
ylabel('PC activity')

%% repeat for total PCA

% PCA of trial-average
[V_TT,L_TT] = eig( cov_total );
[L_TT,sidx] = sort(diag(L_TT),'descend');
V_TT = V_TT(:,sidx);
L_TT = 100*L_TT/sum(L_TT);

% principal components time series
% data are 3D, so we reshape to 2D, project, then reshape back
data2d    = reshape(EEG.data,EEG.nbchan,[]);
comptmp   = V_TT(:,1)'*data2d;
compts_TT = reshape(comptmp,[ EEG.pnts EEG.trials ]);

% then compute the trial average
compTT_erp = mean(compts_TT,2);


% visualize
subplot(321), hold on
plot(L_TT,'ro-','markerfacecolor','w','markersize',8)
legend({'Phase-locked','Total'})

subplot(322), hold on
plot(cumsum(L_TT),'ro-','markerfacecolor','w','markersize',8)
legend({'Phase-locked','Total'})

% topoplot
subplot(324)
##topoplotIndie(V_TT(:,1),EEG.chanlocs,'numcontour',0,'shading','interp','electrodes','off');
topoplotIndieOctave(V_TT(:,1),EEG.chanlocs,'numcontour',0,'shading','interp','electrodes','off');
title('Total c_1')

% time course
subplot(313), hold on
plot(EEG.times,compTT_erp,'linew',2)
legend({'Phase-locked','Total'})

%%
figure
plot(V_TT(:,1),V_PL(:,1),'s')
corr(V_TT(:,1),V_PL(:,1))
