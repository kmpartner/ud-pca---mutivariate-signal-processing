%%
%     COURSE: Dimension reduction and source separation in neuroscience
%    SECTION: Independent components analysis
%      VIDEO: Explore IC distributions in real data
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc


%% import data

load restingstate64chans.mat

% recreate ICA time series
EEG.icaact = (EEG.icaweights*EEG.icasphere)*reshape(EEG.data,EEG.nbchan,[]);

%% show IC topographies and data distributions

% histogram parameters and output matrices
nbins = 500;
[x,y] = deal(zeros(9,nbins));

% let's inspect the first 9 components
for i=1:9
    
    % plot distribution
    figure(1)
    subplot(3,3,i)
    hdat = hist(EEG.icaact(i,:),nbins,'DisplayStyle','stairs');
    set(gca,'xlim',[-1 1]*30)
    title([ 'Component #' num2str(i) ])
    
    % store normalized histograms for later
    y(i,:) = hdat.Values ./ max(hdat.Values);
    x(i,:) = (hdat.BinEdges(1:end-1)+hdat.BinEdges(2:end))/2;
    
    % plot topographical maps
    figure(2)
    subplot(3,3,i)
##    topoplotIndie(EEG.icawinv(:,i),EEG.chanlocs);
##    topoplotIndieOctave(EEG.icawinv(:,i),EEG.chanlocs);
end

%% show all distributions in one plot

figure(3), clf
plot(x',y','linew',1)
set(gca,'xlim',[-150 150]/10)


% for comparison, a Gaussian on top
hold on
h = 5;
t = linspace(-15,15,301);
g = exp( -4*log(2)*t.^2 / h^2 );
plot(t,g,'k--','linew',3)
xlabel('Data values (\muV)')
ylabel('Normalized count')

% to see the tails better, use a log scale
% set(gca,'YScale','log')

%% end.

