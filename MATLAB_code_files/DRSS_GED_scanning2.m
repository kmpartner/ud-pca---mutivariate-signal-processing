%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Source separation with GED
%      VIDEO: Spectral scanning in EEG data
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
% NOTE: don't close figures for comparison with MEG
clear, clc

%%

% For more about this method, see:
% 

%%

% load data
load CEdata.mat
EEG.data = double(EEG.data);

%% parameters and initializations

% specify frequencies in Hz
minFreq =  2;
maxFreq = 30;
numFreq = 35;

frex = linspace(minFreq,maxFreq,numFreq);

% initialize
eigsAll = zeros(numFreq,1);
maps    = zeros(numFreq,EEG.nbchan);

%% broadband covariance

N = EEG.pnts*EEG.trials - 1;

tmpdata = reshape( EEG.data,EEG.nbchan,[] );
tmpdata = bsxfun(@minus,tmpdata,mean(tmpdata,2));
bcov    = tmpdata*tmpdata' / N;

%% find narrowband components


for fi=1:numFreq
    
    % narrowband filter the data
    fdat = filterFGx(EEG.data,EEG.srate,frex(fi),5);
    
    % reshape to 2D
    fdat = reshape(fdat,EEG.nbchan,[]);
    
    % compute S matrix
    fdat = fdat - mean(fdat,2);
    fcov = (fdat*fdat')/(N-1);
    
    % GED
    [evecs,evals] = eig(fcov,bcov);
    [evals,sidx]  = sort(diag(evals),'descend');
    evecs         = evecs(:,sidx);
    
    % save eigenvalues
    eigsAll(fi) = evals(1);
    
    % fix sign and store maps
    tmap       = evecs(:,1)'*fcov;
    [~,maxe]   = max(abs(tmap));
    maps(fi,:) = tmap * sign(tmap(maxe));
end

%% plot

figure(2), clf

% frequencies (in Hz) for topographical inspection
% note: code is currently hard-coded to three frequencies!
frex4topos = [ 4 11 22 ];


subplot(211), hold on
plot(frex,eigsAll,'ko-','markerfacecolor','w','markersize',10,'linew',2)
xlabel('Frequency (Hz)'), ylabel('\lambda')

% plot lines to indicate frequencies
plot(repmat(frex(dsearchn(frex',frex4topos')),2,1),bsxfun(@plus,zeros(2,3),[0 1]'),'k--')


for i=1:3
    % frequency index
    fidx = dsearchn(frex',frex4topos(i));
    
    % topomap
    subplot(2,3,i+3)
##    topoplotIndie(maps(fidx,:),EEG.chanlocs,'numcontour',0);
    topoplotIndieOctave(maps(fidx,:),EEG.chanlocs,'numcontour',0);
    title(sprintf('%.1f Hz',frex(fidx)))
end

%%
