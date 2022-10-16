%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Source separation with GED
%      VIDEO: Cross-validation for regularization
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%% load data

load CEdata.mat
EEG.data = double(EEG.data);

%% get covariance matrices


% filter data in theta
EEG.fdata = filterFGx(EEG.data,EEG.srate,7,4);
filterHz =7;
EEG.fdata = filterFGx(EEG.data,EEG.srate,filterHz,4);

% initialize to zeros
[covS,covR] = deal( zeros(EEG.nbchan) );

% time windows
tidx = dsearchn(EEG.times',[ 0 800 ]');
tidx = dsearchn(EEG.times',[ 0 1250 ]');

for i=1:EEG.trials
    
    % this trial's covariance
    thisTrialCov = cov(squeeze(EEG.fdata(:,tidx(1):tidx(2),i))');
    
##    fdataSqueeze = squeeze(EEG.fdata(:,tidx(1):tidx(2),i));

    % if correct trial
    if i<126
        covR = covR + thisTrialCov;
    else
        covS = covS + thisTrialCov;
    end
end

covR = covR/125;
covS = covS/125;

%% now for cross-validation

% range of gamma parameter values
gammas = linspace(0,.5,43);

% initialize matrices
[maxLambdas,comptsRMS,Renergy] = deal(zeros(size(gammas)));


% loop over gamma values
for gi=1:length(gammas)
    
    % regularize
    g = gammas(gi);
    R = (1-g)*covR + g*mean(eig(covR))*eye(EEG.nbchan);
    
    % GED and sorting index
    [evecs,evals] = eig(covS,R);
    [~,sidx] = sort(diag(evals),'descend');
    
    % component time series
    comp_tsC = evecs(:,sidx(1))'*reshape(EEG.fdata(:,tidx(1):tidx(2),1:125),64,[]);
    comp_tsE = evecs(:,sidx(1))'*reshape(EEG.fdata(:,tidx(1):tidx(2),126:end),64,[]);
    
    % store max eigenvalue and component energy
    maxLambdas(gi) = max(abs(evals(:)));
    comptsRMS(gi) = sqrt(mean(comp_tsE.^2)) / sqrt(mean(comp_tsC.^2));
    Renergy(gi) = norm(R);
end

%% plot the xval results

figure(1), clf
subplot(131)
plot(gammas,maxLambdas,'ks-','markerfacecolor','w','markersize',10)
xlabel('Shrinkage \gamma'), ylabel('\lambda')
title('Maximum eigenvalue')
axis square

subplot(132)
plot(gammas,comptsRMS,'ks-','markerfacecolor','w','markersize',10)
xlabel('Shrinkage \gamma'), ylabel('RMS ratio')
axis square
title('Error:correct RMS')

subplot(133)
plot(gammas,Renergy,'ks-','markerfacecolor','w','markersize',10)
xlabel('Shrinkage \gamma'), ylabel('Norm')
axis square
title('Norm of R')


%% regularized GED

[~,bestGammaLoc] = max(comptsRMS);

% regularization parameter
regu_gam = gammas(bestGammaLoc);

% apply regularization
R = (1-regu_gam)*covR + regu_gam*mean(eig(covR))*eye(EEG.nbchan);


% GED etc
[evecs,evals] = eig(covS,R);
[evals,sidx]  = sort( diag(evals),'descend' );
evecs = evecs(:,sidx);

% get component time series
comp_ts = evecs(:,1)'*reshape(EEG.fdata,64,[]);
comp_ts = reshape(comp_ts,EEG.pnts,EEG.trials);

% power time series
comp_powts = abs(hilbert(comp_ts)).^2;


% plotting
figure(2), clf
subplot(211)
##topoplotIndie(evecs(:,1)'*covS,EEG.chanlocs);
topoplotIndieOctave(evecs(:,1)'*covS,EEG.chanlocs);
title([ '\gamma = ' num2str(regu_gam)])

subplot(212), hold on
plot(EEG.times,mean(comp_powts(:,1:125),2),'k','linew',3)
plot(EEG.times,mean(comp_powts(:,126:end),2),'r','linew',3)
plot([1 1]*EEG.times(tidx(1)),get(gca,'ylim'),'k--')
plot([1 1]*EEG.times(tidx(2)),get(gca,'ylim'),'k--')
set(gca,'xlim',[-200 1000])
legend({'Correct','Error'})
xlabel('Time (ms)'), ylabel('Power')

relpow = mean(mean(comp_powts(tidx(1):tidx(2),126:end))) / mean(mean(comp_powts(tidx(1):tidx(2),1:125)));
title([ 'E:C energy in window: ' num2str(relpow) ])

%%

figure
subplot(231)
imagesc(covS)
colormap
title('covS')

subplot(232)
imagesc(R)
colormap
title('R')

subplot(233)
imagesc(covS-R)
colormap
title('covS - R')

subplot(234)
imagesc(evecs)
colormap
title('evecs')

subplot(235)
plot(1:length(evals), evals(:), '-s')
title('evals')




evalNum = 1

% get component time series
comp_ts = evecs(:,evalNum)'*reshape(EEG.fdata,64,[]);
comp_ts = reshape(comp_ts,EEG.pnts,EEG.trials);

% power time series
comp_powts = abs(hilbert(comp_ts)).^2;


% plotting
figure, clf
subplot(211)
##topoplotIndie(evecs(:,1)'*covS,EEG.chanlocs);
topoplotIndieOctave(evecs(:,evalNum)'*covS,EEG.chanlocs);
topoplotIndieOctave(evecs(:,evalNum)'*covS,EEG.chanlocs, 'numcontour',0,'electrodes','numbers');

title([ '\gamma = ' num2str(regu_gam) ] )

subplot(212), hold on
plot(EEG.times,mean(comp_powts(:,1:125),2),'k','linew',3)
plot(EEG.times,mean(comp_powts(:,126:end),2),'r','linew',3)
plot([1 1]*EEG.times(tidx(1)),get(gca,'ylim'),'k--')
plot([1 1]*EEG.times(tidx(2)),get(gca,'ylim'),'k--')
set(gca,'xlim',[-500 2000])
legend({'Correct','Error'})
xlabel('Time (ms)'), ylabel('Power')

relpow = mean(mean(comp_powts(tidx(1):tidx(2),126:end))) / mean(mean(comp_powts(tidx(1):tidx(2),1:125)));
title([ 'E:C energy in window: ' num2str(relpow)  ', filterHz: ' num2str(filterHz) ', evalNum: ' num2str(evalNum)])
