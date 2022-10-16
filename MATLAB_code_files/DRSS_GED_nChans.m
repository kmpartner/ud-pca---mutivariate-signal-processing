%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Source separation with GED
%      VIDEO: How many electrodes do you need?
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%% preliminaries

% mat file containing EEG, leadfield and channel locations
load emptyEEG256
lf.Gain = double(lf.Gain); % reduced precision for saving
EEG.chanlocs = chanlocs;
EEG.nbchan = length(EEG.chanlocs);

EEG.srate  = 500;
EEG.trials = 100;
EEG.pnts   = 1000; % time points per trial
EEG.times  = (0:EEG.pnts-1)/EEG.srate;
EEG.data   = zeros(EEG.nbchan,EEG.pnts,EEG.trials);

%% dipole locations

dipoleLoc1 = 109;
dipoleLoc2 = 135;


figure(1), clf
clim = [-45 45];
subplot(331)
topoplotIndie(-lf.Gain(:,1,dipoleLoc1), EEG.chanlocs,'maplimits',clim,'numcontour',0,'shading','interp');
title('Simulation dipole 1')

subplot(332)
topoplotIndie(-lf.Gain(:,1,dipoleLoc2), EEG.chanlocs,'maplimits',clim,'numcontour',0,'shading','interp');
title('Simulation dipole 2')

colormap jet

%% insert activity waveforms into dipole data

% frequencies of the two dipoles
freq1 = 15;
freq2 = 10;

% time point of "stimulus" onset
tidx = dsearchn(EEG.times',mean(EEG.times));

% the "innards" of the sine function
omega1 = 2*pi*freq1*EEG.times(tidx:end);
omega2 = 2*pi*freq2*EEG.times(tidx:end);


% loop over trials
for ti=1:EEG.trials
    
    % source waveforms (sine waves with random phase)
    swave1 = sin( omega1 + rand*2*pi );
    swave2 = sin( omega2 + rand*2*pi );
    
    dipole_data = randn(size(lf.Gain,3),EEG.pnts)/5;
    dipole_data(dipoleLoc1,tidx:end) = dipole_data(dipoleLoc1,tidx:end) + swave1;
    dipole_data(dipoleLoc2,tidx:end) = dipole_data(dipoleLoc1,tidx:end) + swave2;
    
    
    % project to scalp
    EEG.data(:,:,ti) = squeeze(lf.Gain(:,1,:))*dipole_data;
end

%% GED for spatial filter

[covPre,covPst] = deal( zeros(EEG.nbchan) );


% covariance matrices per trial
for ti=1:EEG.trials
    
    % "prestim" covariance
    tdat = squeeze(EEG.data(:,1:tidx,ti));
    tdat = bsxfun(@minus,tdat,mean(tdat,2));
    covPre = covPre + (tdat*tdat')/EEG.pnts;
    
    % "post-stim" covariance
    tdat = squeeze(EEG.data(:,tidx:end,ti));
    tdat = bsxfun(@minus,tdat,mean(tdat,2));
    covPst = covPst + (tdat*tdat')/EEG.pnts;
end

covPre = covPre./ti;
covPst = covPst./ti;

% GED
[evecs,evals] = eig(covPst,covPre);
[evals,sidx]  = sort( diag(evals),'descend' );
evecs = evecs(:,sidx);


%%% compute filter forward models and flip sign
% component 1:
maps(:,1) = evecs(:,1)'*covPst; % get component
[~,idx] = max(abs(maps(:,1)));  % find max magnitude
maps(:,1) = maps(:,1)*sign(maps(idx,1)); % possible sign fliip

% component 2:
maps(:,2) = evecs(:,2)'*covPst; % get component
[~,idx] = max(abs(maps(:,2)));  % find max magnitude
maps(:,2) = maps(:,2)*sign(maps(idx,2)); % possible sign fliip


%%% compute component time series (projections)
cdat = evecs(:,1:2)'*reshape(EEG.data,EEG.nbchan,[]);
cdat = reshape( cdat, [ 2 EEG.pnts EEG.trials ]);


%% show topographical maps and eigenspectrum

for i=1:2
    subplot(3,3,3+i)
    topoplotIndie(maps(:,i),EEG.chanlocs,'numcontour',0,'electrodes','off');
    title([ 'Component ' num2str(i) ])
end

subplot(336), plot(evals,'s-','linew',2,'markersize',10,'markerfacecolor','k')
set(gca,'xlim',[0 15])
ylabel('\lambda'), title('Eigenvalues of decomposition'), axis square

%% standard TF analysis on components

% frequencies in Hz
frex = linspace(2,20,20);

% convenient to have component time series data as 2D
comp2d = reshape(cdat,2,[]);

% initialize time-frequency matrix
ctf = zeros(2,length(frex),EEG.pnts);


% loop over frequencies
for fi=1:length(frex)
    
    % filter data for both components at this frequency
    filtdat = filterFGx(comp2d,EEG.srate,frex(fi),4);
    
    % loop over components
    for compi=1:2
        
        % compute power time series as envelope of Hilbert transform
        as = reshape(hilbert(filtdat(compi,:)) ,EEG.pnts,EEG.trials);
        
        % TF power is trial-average power
        ctf(compi,fi,:) = mean( abs(as).^2 ,2);
    end
end

%% some plotting

for compi=1:2
    
    subplot(3,3,6+compi)
    
    contourf(EEG.times,frex,squeeze(ctf(compi,:,:)),40,'linecolor','none')
    
    axis square
    title([ 'Component ' num2str(compi) ])
    xlabel('Time (s)'), ylabel('Frequency (Hz)')
end

%% 



%% now for the channel-count test

% range of channels to subselect
numChannels = round(linspace(10,256,20));

% initialize R2 matrices
[spatR2,timeR2] = deal(zeros(size(numChannels)));

for channum=1:length(numChannels)
    
    %% select subset of channels
    
    chans2use = randperm(size(lf.Gain,1));
    chans2use = chans2use(1:numChannels(channum));
    
    % initialize EEG data
    EEG.nbchan = length(chans2use);
    EEG.data   = zeros(EEG.nbchan,EEG.pnts,EEG.trials);
    
    %% generate dipole data
    
    % initialize matrix to store ground truth time series
    GTts = zeros(EEG.pnts,EEG.trials);
    
    % loop over trials
    for ti=1:EEG.trials
        
        % simulate data and project to scalp
        swave1 = sin( omega1 + rand*2*pi );
        dipole_data = randn(size(lf.Gain,3),EEG.pnts)/5;
        dipole_data(dipoleLoc1,tidx:end) = dipole_data(dipoleLoc1,tidx:end) + swave1;
        EEG.data(:,:,ti) = squeeze(lf.Gain(chans2use,1,:))*dipole_data;
        
        % need to save the dipole time series data on each trial
        GTts(tidx:end,ti) = swave1; % GTts = ground-truth time series
    end
    
    %% compute covariance matrices
    
    [covPre,covPst] = deal( zeros(EEG.nbchan) );
    
    % covariance matrices per trial
    for ti=1:EEG.trials
        
        % "prestim" covariance
        tdat = squeeze(EEG.data(:,1:tidx,ti));
        tdat = bsxfun(@minus,tdat,mean(tdat,2));
        covPre = covPre + (tdat*tdat')/EEG.pnts;
        
        % "post-stim" covariance
        tdat = squeeze(EEG.data(:,tidx:end,ti));
        tdat = bsxfun(@minus,tdat,mean(tdat,2));
        covPst = covPst + (tdat*tdat')/EEG.pnts;
    end
    
    covPre = covPre./ti;
    covPst = covPst./ti;
    
    %% GED
    
    [evecs,evals] = eig(covPst,covPre);
    [evals,sidx]  = sort( diag(evals),'descend' );
    evecs = evecs(:,sidx);
    
    %% compare against ground truth
    
    % spatial correlation (space-R2)
    SFmap = evecs(:,1)'*covPst; % SF = spatial filter
    GTmap = squeeze(lf.Gain(chans2use,1,dipoleLoc1)); % GT = ground truth
    spatR2(channum) = corr(SFmap',GTmap).^2;
    
    % time series correlation (space-R2)
    
    % single-trial data
    compts = evecs(:,1)'*reshape(EEG.data,EEG.nbchan,[]);
    compts = filterFGx( reshape(compts,[EEG.pnts EEG.trials])' ,EEG.srate,freq1,5)';
    timeR2(channum) = corr(compts(:),GTts(:)).^2;
    
end

%% plot

figure(3), clf
subplot(211)
plot(numChannels,spatR2,'s-','linew',2,'markersize',10,'markerfacecolor','k')

subplot(212)
plot(numChannels,timeR2,'s-','linew',2,'markersize',10,'markerfacecolor','k')

%% end
