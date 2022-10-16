%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Source separation with GED
%      VIDEO: Exercise: ZCA+two-stage separation in real data
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%% load data

load sampleEEGdata.mat
EEG.data = double( EEG.data );

%% ZCA

tidx = dsearchn(EEG.times',[-1000 0]');


% covariance of pre-stim activity
tmpdat = reshape( EEG.data(:,tidx(1):tidx(2),:), [ EEG.nbchan (diff(tidx)+1)*EEG.trials ]);
tmpdat = bsxfun(@minus,tmpdat,mean(tmpdat,2));
covN   = tmpdat*tmpdat'/size(tmpdat,2);

% eig and convert to %change
[V,D] = eig(covN);
d = diag(D);
d = 100*d/sum(d);

% threshold and replace data with whitened data
r = d>.1;
EEG.data = reshape( V(:,r)*D(r,r)^(-1/2)*V(:,r)'*reshape(EEG.data,EEG.nbchan,[]) ,size(EEG.data) );


%% Stage 1: compression

% covariance from all broadband data
covA = zeros(EEG.nbchan);
for triali=1:EEG.trials
    tmp = detrend( EEG.data(:,:,triali)' )';
    covA = covA + tmp*tmp'/EEG.pnts;
end

% PCA and sort
[V,D]  = eig(covA/EEG.trials);
[D,sx] = sort(diag(D),'descend');
% sort eigvecs
V = V(:,sx);

% convert evals to %change
D = 100*D/sum(D);

% which components to keep (threshold of % total variance accounted for)
comps2keep = D>1;
##comps2keep = D>0;
nComps = sum(comps2keep);


% copmute subspaced data
EEG.subdata = reshape( (reshape(EEG.data,EEG.nbchan,[])' * V(:,comps2keep))' ,[nComps EEG.pnts EEG.trials]);

%% stage 2 starts here

% filter PC components (compressed data)
EEG.fdata = filterFGx(EEG.subdata,EEG.srate,11,4);


% covariance matrices
[covT,covB] = deal( zeros(nComps) );
for i=1:EEG.trials
    tmp = detrend(squeeze(EEG.subdata(:,:,i))')';
    covB = covB + tmp*tmp'/EEG.pnts;
    
    tmp = detrend(squeeze(EEG.fdata(:,:,i))')';
    covT = covT + tmp*tmp'/EEG.pnts;
end

% GED the shit out that mug, yo.
[evecs,evals] = eig(covT,covB);
[evals,sidx]  = sort( diag(evals),'descend' );

% time series from only the PCA components
ts = reshape( reshape(EEG.subdata,nComps,[])'*evecs(:,sidx(1)) ,EEG.pnts,EEG.trials);

% note the additional transformation using PCA eigenvectors
topo = evecs(:,sidx(1))'*covT*V(:,1:nComps)';
[~,mv] = max(abs(topo));
if topo(mv)<0
    topo = -topo;
    ts = -ts;
end


%% plot the results

figure(1), clf

% topomap
subplot(221)
##topoplotIndie(topo,EEG.chanlocs,'numcontour',0);
topoplotIndieOctave(topo,EEG.chanlocs,'numcontour',0);
title('Filter forward model of top component')

% eigspect
subplot(222)
plot(evals,'s-','markerfacecolor','w','linew',2,'markersize',10)
xlabel('Component number'), ylabel('\lambda')
title('Eigenspectrum')
box off

% time series
subplot(212)
plot(EEG.times,mean(ts,2),'linew',3)
set(gca,'xlim',[-200 1000])
xlabel('Time (ms)')
title('Component time series')
box off

%%
