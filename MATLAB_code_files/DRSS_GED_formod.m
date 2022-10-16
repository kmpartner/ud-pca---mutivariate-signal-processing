%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Source separation with GED
%      VIDEO: GED geometric intuition with covariance surfaces
% Instructor: sincxpress.com
%
%%

%% simulate data

% mat file containing EEG, leadfield and channel locations
load emptyEEG

% index of dipole to simulate activity in
diploc = 109;


% plot brain dipoles
figure(1), clf, subplot(121)
plot3(lf.GridLoc(:,1), lf.GridLoc(:,2), lf.GridLoc(:,3), 'bo','markerfacecolor','y')
hold on
plot3(lf.GridLoc(diploc,1), lf.GridLoc(diploc,2), lf.GridLoc(diploc,3), 'rs','markerfacecolor','k','markersize',10)
rotate3d on, axis square
title('Brain dipole locations')


% Each dipole can be projected onto the scalp using the forward model. 
% The code below shows this projection from one dipole.
subplot(122)
topoplotIndie(-lf.Gain(:,1,diploc), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
title('Signal dipole projection')


% number of time points and time vector
N = 1000;
EEG.times = (0:N-1)/EEG.srate;


% Random data in all brain dipoles
dipole_data = randn(length(lf.Gain),N);

% add signal to second half of dataset
dipole_data(diploc,round(N/2):end) = 15*sin(2*pi*10*EEG.times(round(N/2):end));

% project data from all dipoles to scalp electrodes
EEG.data = squeeze(lf.Gain(:,1,:))*dipole_data;

%% compute GED

% covariance matrix of part1
dataPre = bsxfun(@minus,EEG.data(:,1:round(N/2)-1),mean(EEG.data(:,1:round(N/2)-1),2));
covPre  = dataPre*dataPre'/(N/2);

% covariance matrix of part2
dataPst = bsxfun(@minus,EEG.data(:,round(N/2):end),mean(EEG.data(:,round(N/2):end),2));
covPst  = dataPst*dataPst'/(N/2);


% generalized eigendecomposition
[evecs,evals] = eig( covPst,covPre );

% sort according to eigenvalues
[evals,sidx] = sort(diag(evals),'descend');
evecs = evecs(:,sidx);

% normalize vectors
evecs = bsxfun(@rdivide,evecs,sqrt(sum(evecs.^2,1))); 


% activation map (filter forward model)
actmap = evecs(:,1)'*covPst;


% component time series
comp_ts = evecs(:,1)'*EEG.data;

%% plot results and compare with ground truth

figure(2), clf

% eigenspectrum
subplot(241)
plot(evals(1:20),'ko-','markerfacecolor','w','linew',2)
title('Eigenspectrum'), axis square
ylabel('Percent variance'), xlabel('Component number')

% topographical map of first eigenvector
subplot(242)
topoplotIndie(evecs(:,1),EEG.chanlocs,'numcontour',0,'shading','interp');
title('Weights topomap')

% topographical map of first eigenvector
subplot(243)
topoplotIndie(-actmap,EEG.chanlocs,'numcontour',0,'shading','interp');
title('Filter forward model')

% topographical map of dipole (ground truth)
subplot(244)
topoplotIndie(-lf.Gain(:,1,diploc), EEG.chanlocs,'numcontour',0,'shading','interp');
title('Ground truth topomap')

% plot time series
subplot(212)
plot(EEG.times,-zscore(comp_ts), ...
     EEG.times,zscore(EEG.data(31,:)), ...
     EEG.times,zscore(dipole_data(diploc,:)) ,'linew',2 )
legend({'Comp.';'Chan';'Dipole'})
xlabel('Time (s)')

%%
