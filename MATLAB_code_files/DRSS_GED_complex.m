%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Source separation with GED
%      VIDEO: Complex-valued solutions
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%% load data

load lowrankEEG.mat
## EEG.data = double(EEG.data);

%% rank of matrix

r = rank(EEG.data(:,:,1));
disp("rank = " + r)

%% get covariance matrices

% filter data in alpha
EEG.fdata = filterFGx(EEG.data,EEG.srate,11,4);

% initialize to zeros
[covS,covR] = deal( zeros(EEG.nbchan) );

for i=1:EEG.trials
    
    % detrend instead of demean
    tmp = detrend(squeeze(EEG.data(:,:,i))')';
    covR = covR + tmp*tmp'/EEG.pnts;
    
    tmp = detrend(squeeze(EEG.fdata(:,:,i))')';
    covS = covS + tmp*tmp'/EEG.pnts;
end 

%% regularized GED

% regularization parameter
regu_gam = .0;

% apply regularization
R = (1-regu_gam)*covR + regu_gam*mean(eig(covR))*eye(EEG.nbchan);

% report ranks of matrices
clc
disp("rank(S)  = " + rank(covS))
disp("rank(R)  = " + rank(covR))
disp("rank(Rr) = " + rank(R))


% GED etc
[evecs,evals] = eig(covS,R);
[evals,sidx]  = sort( diag(evals),'descend' );
evecs = evecs(:,sidx);
ts = reshape( reshape(EEG.data,EEG.nbchan,[])'*evecs(:,1) ,EEG.pnts,EEG.trials);

% adjust sign
topo = evecs(:,1)'*covS;
[~,mv] = max(abs(topo));
if topo(mv)<0
    topo = -topo;
    ts = -ts;
end

%% some plotting

figure(1), clf
subplot(211)
##topoplotIndie(topo,EEG.chanlocs);
topoplotIndieOctave(topo,EEG.chanlocs);

subplot(212)
plot(EEG.times,mean(ts,2),'k','linew',3)
set(gca,'xlim',[-200 1000])
xlabel('Time (ms)')

%%
