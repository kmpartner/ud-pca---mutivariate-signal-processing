%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Source separation with GED
%      VIDEO: Task-relevant component in EEG data
% Instructor: sincxpress.com
% 
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%%

% Data were collected in a speeded reaction-time task.
% The first 125 trials are correct responses, the next 125 trials are
% incorrect responses.

load CEdata.mat
EEG.data = double(EEG.data); % higher accuracy with double-precision data

% Apply a narrow band-pass filter centered at 7 Hz with a FWHM of 5 Hz.
% Notice how the results depend on the filter frequency!
fdata = filterFGx(EEG.data,EEG.srate,7,5);
##fdata = filterFGx(EEG.data,EEG.srate,11,5);
##fdata = filterFGx(EEG.data,EEG.srate,7,5, true);


% define time ranges for computing the covariance matrix. You can also try
% different ranges to see how the results are affected.
tidx = dsearchn(EEG.times',[-700 0 700]');
##tidx = dsearchn(EEG.times',[-1500 0 1500]');
##tidx = dsearchn(EEG.times',[-700 0 2000]');


%% GED on pre-stim vs. post-stim covariances

% extract and vectorize data, and compute covariance matrices
datPre = reshape(fdata(:,tidx(1):tidx(2),:),EEG.nbchan,[]);
covPre = cov(datPre');

datPst = reshape(fdata(:,tidx(2):tidx(3),:),EEG.nbchan,[]);
covPst = cov(datPst');


%%% GED
[evecs,evals] = eig( covPst,covPre );
[evals,sidx]  = sort( diag(evals),'descend' );
evecs = evecs(:,sidx);

% initialize maps and time series
maps   = zeros(4,EEG.nbchan);
compts = zeros(4,EEG.pnts,EEG.trials);

%% plotting

figure(1), clf
subplot(211)
plot(evals,'ks-','markersize',10,'markerfacecolor','w')
xlabel('Component')
ylabel('\lambda')


% compute inverse of W (not actually used...)
WinvTranspose = inv(evecs');


for compi=1:4
    
    % compute map and adjust sign
    maps(compi,:) = evecs(:,compi)'*covPst;
    [~,idx]       = max( abs(maps(compi,:)) );
    maps(compi,:) = maps(compi,:) * sign(maps(compi,idx));
    
    % compute time series
    tmpts = evecs(:,compi)'*reshape(EEG.data,EEG.nbchan,[]);
    compts(compi,:,:) = reshape( tmpts,[EEG.pnts EEG.trials] );
    
    subplot(2,4,4+compi)
##    topoplotIndie(maps(compi,:),EEG.chanlocs,'electrodes','off','numcontour',0);
    topoplotIndieOctave(maps(compi,:),EEG.chanlocs,'electrodes','off','numcontour',0);
##    topoplotIndieOctave(maps(compi,:),EEG.chanlocs,'electrodes','numbers','numcontour',0);
    set(gca,'clim',[-1 1]*7)
    title([ 'Component ' num2str(compi) ])
end

colormap jet

%% pick two components from visual inspection


comps2analyze = [ 1 2 ];


%% standard TF analysis on channels

% frequencies in Hz
frex = linspace(1,20,20);

baseidx = dsearchn(EEG.times',[-500 -200]');
t2save  = dsearchn(EEG.times',linspace(-400,1300,80)');

% convenient to have component time series data as 2D
comp2d = reshape(compts(comps2analyze,:,:),2,[]);

% initialize time-frequency matrix
ctf = zeros(2,2,length(frex),length(t2save));


% loop over frequencies
for fi=1:length(frex)
    
    % filter data for both components at this frequency
    filtdat = filterFGx(comp2d,EEG.srate,frex(fi),4);
    
    % loop over components
    for compi=1:2
        
        % compute power time series as envelope of Hilbert transform
        as = reshape(hilbert(filtdat(compi,:)) ,EEG.pnts,EEG.trials);
        
        % trial-average power for each condition
        powrC = mean( abs(as(:,  1:125)).^2 ,2);
        powrE = mean( abs(as(:,126:end)).^2 ,2);
        
        
        % baseline
        base = mean(powrE(baseidx(1):baseidx(2)) + powrC(baseidx(1):baseidx(2)))/2;
        
        % TF power is trial-average power
        ctf(compi,1,fi,:) = 10*log10( powrC(t2save)/base );
        ctf(compi,2,fi,:) = 10*log10( powrE(t2save)/base );
    end
end

%% some plotting

figure(2), clf

clim = [-1 1]*10;
condlabels = {'Correct';'Error'};

for compi=1:2
    for condi=1:2
        
        subplot(2,2,(compi-1)*2+condi)
        
        contourf(EEG.times(t2save),frex,squeeze(ctf(compi,condi,:,:)),40,'linecolor','none')
        
        axis square
        set(gca,'xlim',EEG.times(t2save([1 end])),'clim',clim)
        title([ 'Component ' num2str(comps2analyze(compi)) ', ' condlabels{condi} ])
        xlabel('Time (s)'), ylabel('Frequency (Hz)')
    end
end

colormap jet

%% done.


figure
subplot(2,3,1)
imagesc(covPre)
title('covPre')
colorbar

subplot(2,3,2)
imagesc(covPst)
caxis([0 50])
title('covPst')
colorbar

##subplot(2,3,3)
##imagesc(covPst - covPre)
####caxis([-100000, 100000])
##title('covPst - covPre')
##colorbar

[evecs2,evals2] = eig(covPst,covPre);


subplot(2,2,3)
imagesc(evecs2)
colorbar
title('evecs')

subplot(2,2,4)
imagesc(evals2)
colorbar
title('evals')
