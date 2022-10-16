%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Source separation with GED
%      VIDEO: Effects of regularization in real data
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%% 

% the transformation matrix T
T = eye(3);
T(end) = 0;

% create the spherical coordinates
[x,y,z] = sphere;
M = [x(:) y(:) z(:)]';

%%% setup the figure
figure(1), clf
h = surf(x,y,z);
shading interp
axis([-1 1 -1 1 -1 1])
axis square, rotate3d on
view([-70 15])


% loop over gamma values
for gami=linspace(0,.5,20)

    Tt = T*(1-gami) + gami*mean(eig(T))*eye(3);
    
    % impose the transformation matrix onto the sphere coordinates
    TM = Tt*M;
    
    % reshape back to new coordinatess
    xp = reshape(TM(1,:),[21 21]); % note: 21 is default resolution for sphere function
    yp = reshape(TM(2,:),[21 21]);
    zp = reshape(TM(3,:),[21 21]);
    
    % update the plot
    set(h,'XData',xp,'YData',yp,'ZData',zp)
    title([ 'Regularization \gamma = ' num2str(gami) ])
    pause(.3)
end

%%

%% now for real data...

load sampleEEGdata.mat

% regularization parameter
regu_gam = .0;
##regu_gam = .01;
##regu_gam = .001;

% filter data in alpha
EEG.fdata = filterFGx(EEG.data,EEG.srate,11,4);

% get covariances from each trial
[covF,covB] = deal( zeros(EEG.nbchan) );

for i=1:EEG.trials
    tmp  = detrend(squeeze(EEG.data(:,:,i))')';
    covB = covB + tmp*tmp'/EEG.pnts;
    
    tmp  = detrend(squeeze(EEG.fdata(:,:,i))')';
    covF = covF + tmp*tmp'/EEG.pnts;
end 

covB = covB/i;
covF = covF/i;


% apply regularization
Rregu = (1-regu_gam)*covB + regu_gam*mean(eig(covB))*eye(EEG.nbchan);

% GED
[evecs,evals] = eig(covF,Rregu);
[evals,sidx]  = sort( diag(evals),'descend' );
evecs = evecs(:,sidx);
ts = reshape( reshape(EEG.data,EEG.nbchan,[])'*evecs(:,1) ,EEG.pnts,EEG.trials);

% topomaps and time series
topo = evecs(:,1)'*covF;
[~,mv] = max(abs(topo));
if topo(mv)<0
    topo = -topo;
    ts = -ts;
end

figure(3)
subplot(332)
##topoplotIndie(topo,EEG.chanlocs);
topoplotIndieOctave(topo,EEG.chanlocs);

subplot(312)
plot(EEG.times,mean(ts,2),'k','linew',3)
set(gca,'xlim',[-200 1000])



%%% show both covariance matrices
subplot(3,3,7)
imagesc(covB)
axis square
set(gca,'xtick',[],'ytick',[],'clim',[-1 1]*150)
title('R matrix')
colorbar

subplot(3,3,8)
imagesc(Rregu)
set(gca,'xtick',[],'ytick',[],'clim',[-1 1]*150)
axis square
title([ 'Regularized (\gamma=' num2str(100*regu_gam) '%)' ])
colorbar

subplot(3,3,9)
imagesc(covB-Rregu)
set(gca,'xtick',[],'ytick',[],'clim',[-1 1]*10)
axis square
title('R - Rregu')
colorbar
colormap jet


%% done.
