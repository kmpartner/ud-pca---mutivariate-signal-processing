%%
%     COURSE: PCA & multivariate signal processing, applied to neural data
%    SECTION: Overfitting and inferential statistics
%      VIDEO: MATLAB: Cross-validation in real data
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%%


% load data
load CEdata.mat
EEG.data = double(EEG.data);

% narrowband filter the data
filterCent = 7;
EEG.fdata = filterFGx(EEG.data,EEG.srate,filterCent,3);

% limited time window for covariance
tidx = dsearchn(EEG.times',[0 700]');

% define blocks of trials for training/testing
numblocks = 10;
trialblocks = ceil( numblocks*(1:125)/125);

% initialize output matrices
maxevals = zeros(numblocks,1);
map      = zeros(numblocks,EEG.nbchan);
powts    = zeros(numblocks,2,EEG.pnts);


% loop over blocks
for blocki=1:numblocks
    
    %%% identify the training/testing trials
    training_trialsE = 125 + find(trialblocks~=blocki);
    training_trialsC =       find(trialblocks~=blocki);
    
    testing_trialsE  = 125 + find(trialblocks==blocki);
    testing_trialsC  =       find(trialblocks==blocki);
    
    
    %%% subdata for only this block of trials
    dataE = EEG.fdata(:,tidx(1):tidx(2),training_trialsE);
    dataE = reshape(dataE,EEG.nbchan,[]);
    covE  = cov(dataE');
    
    dataC = EEG.fdata(:,tidx(1):tidx(2),training_trialsC);
    dataC = reshape(dataC,EEG.nbchan,[]);
    covC  = cov(dataC');
    
    
    %%% fit GED to all-but-one blocks
    [evecs,evals] = eig(covE,covC);
    [evals,sidx]  = sort(diag(evals),'descend');
    evecs = evecs(:,sidx);
    
    
    %%% save eig outputs
    maxevals(blocki) = evals(1);
    tmpmap = evecs(:,1)'*covE;
    
    % force map sign
    [~,mapmax] = max(abs(tmpmap));
    map(blocki,:) = tmpmap*sign(tmpmap(mapmax));
    
    
    
    %%% component power time series FROM TESTING DATA
    compfilt = evecs(:,1)'*reshape(EEG.data,EEG.nbchan,[]);
    compfilt = filterFGx(compfilt,EEG.srate,filterCent,5);
    comppow  = abs(hilbert( compfilt )).^2;
    comppow  = reshape(comppow,EEG.pnts,EEG.trials);
    
    powts(blocki,1,:) = mean(comppow(:,testing_trialsC),2);
    powts(blocki,2,:) = mean(comppow(:,testing_trialsE),2);
end


%% show results

% Note: A measure of "goodness" is the coefficient of variation (CV).
%       CV is std/mean. Small values are good, large values indicate large
%       uncertainty or lack of precision.

figure(1), clf
subplot(221)
plot(maxevals,'ks','markersize',13,'markerfacecolor','r')
title([ '\lambda''s CV: ' num2str( std(maxevals)/mean(maxevals) ) ])
xlabel('Train-test block'), ylabel('Max-\lambda')


% average filter forward model
subplot(222)
##topoplotIndie(mean(map),EEG.chanlocs);
topoplotIndieOctave(mean(map),EEG.chanlocs);
title('Average filter forward model')


% power time courses (first compute standard error and average across blocks).
ste_pow = squeeze(std(powts,[],1)) ./ sqrt(numblocks-1);
ave_pow = squeeze(mean(powts));


% show in patches 
subplot(212), hold on
h = patch([EEG.times EEG.times(end:-1:1)],[ave_pow(1,:)+2*ste_pow(1,:) ave_pow(1,end:-1:1)-2*ste_pow(1,end:-1:1)],'b');
set(h,'facealpha',.2,'edgecolor','none')

h = patch([EEG.times EEG.times(end:-1:1)],[ave_pow(2,:)+2*ste_pow(2,:) ave_pow(2,end:-1:1)-2*ste_pow(2,end:-1:1)],'r');
set(h,'facealpha',.2,'edgecolor','none')

% and average on top
plot(EEG.times,squeeze(mean(powts,1)),'linew',3)

set(gca,'xlim',[-500 1500])
legend({'Correct';'Error'})
title([ 'Power time series at ' num2str(filterCent) ' Hz' ])
xlabel('Time (ms)'), ylabel('Power')

%% maps from all folds

% try this code:
figure(2)
for i=1:10
    subplot(3,4,i)
##    topoplotIndie(map(i,:),EEG.chanlocs,'electrodes','off');
    topoplotIndieOctave(map(i,:),EEG.chanlocs,'electrodes','off');
    title("Map from fold " + i)
end

%% done.
