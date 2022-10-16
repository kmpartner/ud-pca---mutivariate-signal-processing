%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Source separation with GED
%      VIDEO: Simulated data with and without ZCA
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
clear, clc


%% preliminaries

dozca = true; % true or false
##dozca = false;

% mat file containing EEG, leadfield and channel locations
load emptyEEG
EEG.srate = 500;

EEG.trials = 200;  % total, 1/2 per condition
EEG.pnts   = 1000; % time points per trial
EEG.times  = (0:EEG.pnts-1)/EEG.srate;
EEG.data   = zeros(EEG.nbchan,EEG.pnts,EEG.trials);

%% dipole locations

dipoleLoc1 = 109;


figure(dozca+1), clf
clim = [-45 45];
subplot(221)
##topoplotIndie(-lf.Gain(:,2,dipoleLoc1), EEG.chanlocs,'maplimits',clim,'numcontour',0,'electrodes','numbers','shading','interp');
topoplotIndieOctave(-lf.Gain(:,2,dipoleLoc1), EEG.chanlocs,'maplimits',clim,'numcontour',0,'electrodes','numbers','shading','interp');
title('Simulation dipole 1')

colormap jet

%% insert activity waveforms into dipole data

% frequency of the dipole
freq1 = 15;

% time point of "stimulus" onset
tidx = dsearchn(EEG.times',mean(EEG.times));

% the "innards" of the sine function
omega1 = 2*pi*freq1*EEG.times(tidx:end);


% loop over trials
for ti=1:EEG.trials
    
    % source waveforms (sine waves with random phase)
    swave1 = sin( omega1 + rand*2*pi );
    
    dipole_data = randn(size(lf.Gain,3),EEG.pnts)/5;
    dipole_data(dipoleLoc1,tidx:end) = dipole_data(dipoleLoc1,tidx:end) + swave1;
    
    % project to scalp
    EEG.data(:,:,ti) = squeeze(lf.Gain(:,2,:))*dipole_data;
end

%% ZCA

% PCA
tmpdat = reshape( EEG.data,EEG.nbchan,[] );
tmpdat = tmpdat - mean(tmpdat,2);
covmat = tmpdat*tmpdat' / (length(tmpdat)-1);
[V,D]  = eig(covmat);

% ZCA (check yz*yz'!)
yz = V*D^(-1/2)*V'*reshape(EEG.data,EEG.nbchan,[]);


%%% replace data with whitened data
if dozca
    EEG.data = reshape( yz,size(EEG.data) );
end

%% GED for spatial filter

[covPre,covPst] = deal( zeros(EEG.nbchan) );


% covariance matrices per trial
for ti=1:EEG.trials
    
    % "prestim" covariance
    tdat = squeeze(EEG.data(:,1:tidx,ti));
    tdat = tdat - mean(tdat,2);
    covPre = covPre + (tdat*tdat')/(EEG.pnts/2);
    
    % "post-stim" covariance
    tdat = squeeze(EEG.data(:,tidx:end,ti));
    tdat = tdat - mean(tdat,2);
    covPst = covPst + (tdat*tdat')/(EEG.pnts/2);
end

covPre = covPre./ti;
covPst = covPst./ti;

% GED
[evecs,evals] = eig(covPst,covPre);
[evals,sidx]  = sort( diag(evals),'descend' );
evecs = evecs(:,sidx);


%%% compute filter forward models and flip sign
maps = evecs(:,1)'*covPst;   % get component
[~,idx] = max(abs(maps));    % find max magnitude
maps = maps*sign(maps(idx)); % possible sign flip


%%% compute component time series (projections)
cdat = reshape( evecs(:,1)'*reshape(EEG.data,EEG.nbchan,[]), [ EEG.pnts EEG.trials ]);


%% show topographical maps and eigenspectrum

subplot(223)
##topoplotIndie(maps,EEG.chanlocs,'numcontour',0,'electrodes','off');
topoplotIndieOctave(maps,EEG.chanlocs,'numcontour',0,'electrodes','off');
title('Component map')

subplot(222), plot(evals,'s-','linew',2,'markersize',10,'markerfacecolor','k')
set(gca,'xlim',[0 15])
ylabel('\lambda'), title('Eigenvalues of decomposition'), axis square

%% quick TF analysis on component time series

% frequencies in Hz
frex = linspace(2,20,20);

% convenient to have component time series data as 2D
comp2d = reshape(cdat,1,[]);

% initialize time-frequency matrix
ctf = zeros(length(frex),EEG.pnts);


% loop over frequencies
for fi=1:length(frex)
    
    % filter data for both components at this frequency
    filtdat = filterFGx(comp2d,EEG.srate,frex(fi),4);
    
    % compute power time series as envelope of Hilbert transform
    as = reshape(hilbert(filtdat) ,EEG.pnts,EEG.trials);
    
    % TF power is trial-average power
    ctf(fi,:) = mean( abs(as).^2 ,2);
end

%% some plotting

subplot(224)
contourf(EEG.times,frex,ctf,40,'linecolor','none')
axis square
xlabel('Time (s)'), ylabel('Frequency (Hz)')

%% end
