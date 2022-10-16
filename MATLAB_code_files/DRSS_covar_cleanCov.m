%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Creating and interpreting covariance matrices
%      VIDEO: MATLAB: Single trial covariance distances
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%%

% load data
load sampleEEGdata

%% compute covariances and distances

% compute average of single-trial covariances
% Without storing individual covmats!
covave = zeros( EEG.nbchan );
for triali=1:EEG.trials
    covave = covave + cov( squeeze(EEG.data(:,:,triali))' );
end

% divide by number of trials
covave = covave / triali;



% now loop through trials and compute the distance to the average
covdist = zeros(EEG.trials,1);

for triali=1:EEG.trials
    thistrialcov = cov( squeeze(EEG.data(:,:,triali))' );
    
    % compute Frobenius distance
    covdist(triali) = sqrt( sum(thistrialcov(:) .* covave(:)) );
    % previous line is the same as this one:
    %covdist(triali) = sqrt( trace(thistrialcov'*covave) );
    
    % alternative: Euclidean distance (gives similiar results)
    %covdist(triali) = sqrt( sum((thistrialcov(:) - covave(:)).^2) );
end

% convert to z
covdistz = (covdist-mean(covdist)) / std(covdist);

%% visual inspection of covariance distances to average

% show the covariance distances
figure(1), clf
subplot(2,3,1:2)
plot(covdistz,'ks-','linew',2,'markerfacecolor','w','markersize',12)
xlabel('Trial'), ylabel('Z_{dist}')
title('Z-scored covariance distances')


% histogram of distances
subplot(233)
hist(covdistz,10)
xlabel('Distances'), ylabel('Count')
title('Histogram of distances')


%% pick a threshold and reject trials

% threshold
thresh = 2.3; % ~.01

% identify trials that exceed the threshold
toofar = covdistz>thresh;

% remove those trials from the data
data2 = EEG.data;
data2(:,:,toofar) = [];

% check sizes

%% show some data before and after rejection

% plot time courses
subplot(212), hold on
plot(EEG.times,mean(EEG.data(31,:,:),3),'k','linew',2)
plot(EEG.times,mean(data2(31,:,:),3),'r','linew',2)

% make the plot look a bit nicer
xlabel('Time (a.u.)')
legend({'Original data';'Trials removed'})
title('Time series before and after covariance cleaning')
zoom on

%%

figure
imagesc(covave)

