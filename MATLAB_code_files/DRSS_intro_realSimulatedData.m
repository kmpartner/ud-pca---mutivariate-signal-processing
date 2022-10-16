%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Introductions
%      VIDEO: Using real and simulated data
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%% load and explore the data

load ../MATLAB_data_files/sampleEEGdata.mat
whos

%% creating topographical maps

% parameters to select data
timeidx  = dsearchn(EEG.times',230);
trialidx = 10;

% grab one time slice
data = EEG.data(:,timeidx,trialidx);

% visualize
figure(1), clf
topoplotIndie( data, EEG.chanlocs );
colorbar
title([ 'Topography at ' num2str(EEG.times(timeidx),'%.0f') ' ms, trial ' num2str(trialidx) ])

%% simulate EEG data

% mat file containing EEG, leadfield and channel locations
clear
load ../MATLAB_data_files/emptyEEG
whos

%% pick and visualize a dipole and its projection

% index of dipole to simulate activity in
diploc = 109;


% plot brain dipoles
figure(2), clf
subplot(221)
plot3(lf.GridLoc(:,1), lf.GridLoc(:,2), lf.GridLoc(:,3), 'bo','markerfacecolor','y')
hold on
plot3(lf.GridLoc(diploc,1), lf.GridLoc(diploc,2), lf.GridLoc(diploc,3), 'rs','markerfacecolor','k','markersize',10)
rotate3d on, axis square
title('Brain dipole locations')


% Each dipole can be projected onto the scalp using the forward model. 
% The code below shows this projection from one dipole.
subplot(222)
topoplotIndie(-lf.Gain(:,1,diploc), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
title('Signal dipole projection')

%% create time series data

% number of time points and time vector
N = 1000;
EEG.times = (0:N-1)/EEG.srate;


% Random data in all brain dipoles
dipole_data = randn(length(lf.Gain),N);

% add signal to one dipole
dipole_data(diploc,:) = 15*sin(2*pi*10*EEG.times);

% project data from all dipoles to scalp electrodes
EEG.data = squeeze(lf.Gain(:,1,:))*dipole_data;


%% plot the time series data

subplot(212)
plot(EEG.times,EEG.data(31,:), ...
     EEG.times,-dipole_data(diploc,:)*100 ,'linew',2 )
legend({'Chan';'Dipole'})
xlabel('Time (s)')

%% done.
