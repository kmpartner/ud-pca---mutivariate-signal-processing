%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Source separation with GED
%      VIDEO: MATLAB: Simulated EEG data with alternating dipoles
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%% setup

% mat file containing EEG, leadfield and channel locations
load emptyEEG

% simulation parameters
EEG.srate  = 500; % Hz
EEG.trials = 100;
EEG.pnts   = 1000;
EEG.times  = (0:EEG.pnts-1)/EEG.srate;

% scale up grid locations (convenience)
lf.GridLoc = lf.GridLoc*100;

nDipoles = 8; % number of distinct dipole locations
##nDipoles = 1; % number of distinct dipole locations

% normal dipoles (normal to the surface of the cortex)
lf.GainN = bsxfun(@times,squeeze(lf.Gain(:,1,:)),lf.GridOrient(:,1)') + bsxfun(@times,squeeze(lf.Gain(:,2,:)),lf.GridOrient(:,2)') + bsxfun(@times,squeeze(lf.Gain(:,3,:)),lf.GridOrient(:,3)');

%% pick dipoles 

% index of starting dipole location
dipoleLoc = 109;

% distances to starting dipole location
locdiffs = lf.GridLoc - lf.GridLoc(dipoleLoc,:);

% pick some dipole locations for the simulation
locs2use = sort(find( abs(locdiffs(:,2))<.5 & abs(locdiffs(:,3))<.5 ));

for i=1:length(locs2use)
    zd     = lf.GridLoc - lf.GridLoc(locs2use(i),:);
    ttu    = find(abs(zd(:,2))<.5 & abs(zd(:,1))<.5);
    [~,le] = max( lf.GridLoc(ttu,3) );
    
    locs2use(i) = ttu(le);
end

locs2use = locs2use(2:2:end);
locs2use = locs2use(1:nDipoles);

%% plot the dipole locations and forward projections

% in a brain plot
figure(1), clf
plot3(lf.GridLoc(:,1),lf.GridLoc(:,2),lf.GridLoc(:,3),'ko'), hold on
plot3(lf.GridLoc(locs2use,1),lf.GridLoc(locs2use,2),lf.GridLoc(locs2use,3),'bo','markerfacecolor','b','markersize',10)
axis square, rotate3d on

% plot topographies
figure(2), clf
clim = [-30 30];
for i=1:nDipoles
    subplot(3,3,i)
##    topoplotIndie(lf.GainN(:,locs2use(i)), EEG.chanlocs,'numcontour',0,'electrodes','off');
    topoplotIndieOctave(lf.GainN(:,locs2use(i)), EEG.chanlocs,'numcontour',0,'electrodes','off');
    set(gca,'clim',clim)
end

subplot(339)
##topoplotIndie(mean(lf.GainN(:,locs2use),2), EEG.chanlocs,'numcontour',0,'electrodes','off');
topoplotIndieOctave(mean(lf.GainN(:,locs2use),2), EEG.chanlocs,'numcontour',0,'electrodes','off');
set(gca,'clim',clim)
title('Average')

%% create condition wave

% frequency of the dipole
freq1 = 15;

% the "task activity"
taskwave = sin( 2*pi*freq1*EEG.times );

% center time points for windowing each dipole's activation function
centtimes = EEG.times( round( linspace(EEG.pnts*.1,EEG.pnts*.9,9) ));

% trials in which the waveform should be included
taskTrials = round(EEG.trials/2):EEG.trials;

%% create data

% initialize
EEG.data  = zeros(EEG.nbchan,EEG.pnts,EEG.trials);
sourceact = zeros(nDipoles,EEG.pnts,EEG.trials);


% loop over trials
for triali=1:EEG.trials
    
    % brain of noise
    dipdat = .2*randn(size(lf.GridLoc,1),EEG.pnts);
    
    % insert task wave only in 2nd half of trials
    if ismember(triali,taskTrials)
        
        % loop through each dipole and set its activity 
        % as a delayed/windowed sine wave
        for li=1:nDipoles
            dipdat(locs2use(li),:) = dipdat(locs2use(li),:) + taskwave .* exp( -(EEG.times-centtimes(li)).^2/.025);
        end
    end
    
    % project to scalp
    EEG.data(:,:,triali) = lf.GainN*dipdat;
    
    % store dipole activation for later comparison (on your own!)
    sourceact(:,:,triali) = dipdat(locs2use,:);
end

%% compare covariance matrices around center time points

% width of time window (in points = 400 ms window around center)
npnts = 99;

% initialize covariances matrix
covmats = zeros(nDipoles,EEG.nbchan,EEG.nbchan);

clim = [-1 1]*1e5;

figure(3), clf
for wini=1:nDipoles
    
    % time window
    cidx = dsearchn(EEG.times',centtimes(wini));
    
    % data around this center time point
    tmpdat = reshape( EEG.data(:,cidx-npnts:cidx+npnts,taskTrials),EEG.nbchan,[] );
    tmpdat = bsxfun(@minus,tmpdat,mean(tmpdat,2));
    
    % covariance matrix
    covmats(wini,:,:) = tmpdat*tmpdat' / size(tmpdat,2);
    
    
    % now show the matrix
    subplot(3,3,wini)
    imagesc(squeeze(covmats(wini,:,:)))
    axis square
    title([ 'Covmat, time ' num2str(EEG.times(cidx)*1000) ' ms' ])
    set(gca,'xtick',[],'ytick',[],'clim',clim)
end

% average covariance matrix
subplot(339)
imagesc(squeeze(mean(covmats,1)))
axis square
set(gca,'xtick',[],'ytick',[],'clim',clim)
title('Average')


%% GED for spatial filter

[cov1,cov2] = deal( zeros(EEG.nbchan) );

% covariance matrices
for triali=1:EEG.trials
    
    % data from this trial
    tdat = squeeze(EEG.data(:,:,triali));
    tdat = tdat - mean(tdat,2);
    
    % add to baseline or task covariance matrix
    if ismember(triali,taskTrials)
        cov2 = cov2 + (tdat*tdat')/EEG.pnts;
    else
        cov1 = cov1 + (tdat*tdat')/EEG.pnts;
    end
end

cov1 = cov1./triali/2;
cov2 = cov2./triali/2;

% GED here...
[evecs,evals] = eig( cov2,cov1 );
[evals,sidx]  = sort( diag(evals),'descend' );
evecs = evecs(:,sidx);
maps  = evecs(:,1:3)'*cov2; % keep top three components

% show the eigenspectrum
figure(4), clf
plot(evals,'ko-','linew',2,'markerfacecolor','w','markersize',10)
xlabel('Component number'), ylabel('\lambda')
title('Eigenspectrum')

%%


% component data from three components
cdat = reshape( evecs(:,1:3)'*reshape(EEG.data,EEG.nbchan,[]), [3 EEG.pnts,EEG.trials]);


% plot topomaps of components
figure(5), clf
subplot(241), 
##topoplotIndie(mean(lf.GainN(:,locs2use),2),EEG.chanlocs,'electrodes','off','numcontour',0);
topoplotIndieOctave(mean(lf.GainN(:,locs2use),2),EEG.chanlocs,'electrodes','off','numcontour',0);
 title({'Average dipole projections';'from simulation'})
 
subplot(242), 
##topoplotIndie(maps(1,:),EEG.chanlocs,'electrodes','off','numcontour',0);
topoplotIndieOctave(maps(1,:),EEG.chanlocs,'electrodes','off','numcontour',0);
 title('1st component')
 
subplot(243), 
##topoplotIndie(maps(2,:),EEG.chanlocs,'electrodes','off','numcontour',0);
topoplotIndieOctave(maps(2,:),EEG.chanlocs,'electrodes','off','numcontour',0);
 title('2nd component')
 
subplot(244), 
##topoplotIndie(maps(3,:),EEG.chanlocs,'electrodes','off','numcontour',0);
topoplotIndieOctave(maps(3,:),EEG.chanlocs,'electrodes','off','numcontour',0);
 title('3rd component')

colormap jet

subplot(2,2,[3 4])

% get time series
ampts = zeros(size(cdat,1),EEG.pnts);
for ci=1:size(cdat,1)
    fdat = filterFGx(squeeze(cdat(ci,:,taskTrials))',EEG.srate,freq1,4);
    hdat = hilbert( fdat' );
    adat = abs( hdat );
    ampts(ci,:) = mean(adat,2);
end

plot(EEG.times,ampts,'linew',3)
legend({'Comp1';'Comp2';'Comp3'})
xlabel('Time (sec.)'), ylabel('Activation')
title('Component amplitude time series')

%% 
