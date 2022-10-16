%%
%     COURSE: Dimension reduction and source separation in neuroscience
%    SECTION: Source separation for steady-state responses
%      VIDEO: MATLAB: Example with real EEG data
% Instructor: mikexcohen.com
%
%%

clear, clc

load SSVEPdata.mat
EEG.data = double(EEG.data);

%% specify RESS parameters

% RESS parameters
ssvepfrex = [ 15 20   24 17.14 ];
neig      = 1;  % distance to frequency neighbors in Hz
fwhm_targ = .5; % FWHM in Hz for target
fwhm_neig = 1;  % FWHM in Hz for neighbors

% shrinkage proportion
shr = .01;

% time window for SSVEP
tidx = dsearchn(EEG.times',[ -500 1300 ]');

% number of time points in filter
pnts = (diff(tidx)+1)*EEG.trials;


% FFT param
nfft = ceil( EEG.srate/.05 );
hz   = linspace(0,EEG.srate,nfft);

%% test RESS temporal filter parameters

for fi=1:length(ssvepfrex)
    
    % filter parameters
    frex = [ ssvepfrex(fi)-neig ssvepfrex(fi) ssvepfrex(fi)+neig ];
    stds = [ fwhm_neig fwhm_targ fwhm_neig ];
    
    RESSfilterFGxTest(pnts,EEG.srate,frex,stds,100+fi);
end

%% now for RESS

figure(1), clf

for fi=1:length(ssvepfrex)
    
    %% S covariance matrix
    
    % filter
    data = filterFGx(EEG.data,EEG.srate,ssvepfrex(fi),fwhm_targ);
    
    % extract and mean-center data
    data = data(:,tidx(1):tidx(2),:);
    data = reshape( data ,EEG.nbchan,[] );
    data = data - mean(data,2);
    
    % covariance
    covS = (data*data') / (pnts-1);
    
    %% R covariance matrices
    
    % lower R/2
    data = filterFGx(EEG.data,EEG.srate,ssvepfrex(fi)-neig,fwhm_neig);
    
    % extract and mean-center data
    data = data(:,tidx(1):tidx(2),:);
    data = reshape( data ,EEG.nbchan,[] );
    data = data - mean(data,2);
    
    % covariance
    covRl = (data*data') / (pnts-1);
    
    
    % upper R/2
    data = filterFGx(EEG.data,EEG.srate,ssvepfrex(fi)+neig,fwhm_neig);
    
    % extract and mean-center data
    data = data(:,tidx(1):tidx(2),:);
    data = reshape( data ,EEG.nbchan,[] );
    data = data - mean(data,2);
    
    % covariance
    covRu = (data*data') / (pnts-1);
    
    % full R matrix is average of lower/upper
    covR = (covRl+covRu)/2;
    
    %% GED with optional shrinkage
    
    % apply shrinkage to covR
    covR = (1-shr)*covR + shr*mean(eig(covR))*eye(size(covR));
    
    % GED and sort components
    [evecs,evals] = eig(covS,covR);
    [evals,sidx]  = sort(diag(evals),'descend');
    evecs = evecs(:,sidx);
    
    % compute filter forward model and flip sign
    map = evecs(:,1)'*covS;
    [~,maxchan] = max(abs(map));
    map = map*sign(map(maxchan));
    
    
    %% component time series
    
    % compute component time series
    compts = evecs(:,1)'*reshape(EEG.data,EEG.nbchan,[]);
    compts = reshape(compts,EEG.pnts,EEG.trials);
    
    % power spectrum averaged over trials
    powr = mean( abs(fft(compts(tidx(1):tidx(2),:),nfft,1)/EEG.pnts).^2 ,2);
    
    
    % SNR spectrum
    skipbins =  5; % .5 Hz
    numbins  = 20+skipbins; % 2 Hz
    snrSpect = zeros(size(powr));
    
    for hzi=numbins+1:length(hz)-numbins-1
        
        % SNR over all time points and conditions
        numer = powr(hzi);
        denom = mean(powr([hzi-numbins:hzi-skipbins hzi+skipbins:hzi+numbins]) );
        snrSpect(hzi) = numer./denom;
    end        
    
    %% plotting
    
    subplot(4,4,fi*4-3)
    plot(evals(1:20),'s-','markerfacecolor','k','linew',2)
    xlabel('Component'), ylabel('eigenspectrum')
    title([ num2str(ssvepfrex(fi)) ' Hz' ])
    
    subplot(4,4,fi*4-2)
##    topoplotIndie(map,EEG.chanlocs,'numcontour',0,'electrodes','off');
    topoplotIndieOctave(map,EEG.chanlocs,'numcontour',0,'electrodes','off');
    title([ num2str(ssvepfrex(fi)) ' Hz' ])
    
    subplot(4,4,fi*4-1)
    plot(hz,powr,'k','linew',2)
    set(gca,'xlim',[min(ssvepfrex)-5 max(ssvepfrex+5)])
    xlabel('Frequency (Hz)'), ylabel('Raw power')
    title([ num2str(ssvepfrex(fi)) ' Hz' ])
    
    subplot(4,4,fi*4)
    plot(hz,snrSpect,'k','linew',2)
    set(gca,'xlim',[min(ssvepfrex)-5 max(ssvepfrex+5)])
    xlabel('Frequency (Hz)'), ylabel('SNR')
    title([ num2str(ssvepfrex(fi)) ' Hz' ])
    
    
end

%%
