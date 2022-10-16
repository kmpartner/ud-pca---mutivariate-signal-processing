%%
%     COURSE: Dimension reduction and source separation in neuroscience
%    SECTION: Independent components analysis
%      VIDEO: Data distributions and ICA
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc


%% 


% a number
N = 1000;

% a non-Gaussian distribution
dist1 = rand(N,1);

% another non-Gaussian distribution
dist2 = rand(N,1).^2;

% their separate and combined distributions
figure(1), clf
subplot(221)
hist(dist1,100)
title('Distribution 1')

subplot(222)
hist(dist2,100)
title('Distribution 1')

subplot(212)
hist(dist1+dist2,100)
title('Distribution 1+2')

%%% QUESTION: What is the shape of the combined distribution? 
%             Are you surprised at the result?
%             What does this tell you about mixing distributions?

%% ICA on a trivial example (unmixed data)

clear data

% two non-Gaussian distributions
data(1,:) = rand(1,N);
data(2,:) = rand(1,N).^4;


% ICA and scores
b = jader(data);
iscores = b*data;


% plot distributions
figure(2), clf
% data 1
subplot(221), 
hist(data(1,:),100)
title('Data 1')

% data 2
subplot(222), 
hist(data(2,:),100)
title('Data 2')

% IC 1
subplot(223), 
hist(iscores(1,:),100)
title('IC 1')

% IC 2
subplot(224), 
hist(iscores(2,:),100)
title('IC 2')



% now plot data as a function of ICs
figure(3), clf
subplot(121)
plot(data(1,:),iscores(1,:),'o')
xlabel('Data'), ylabel('IC1 scores')
axis square

subplot(122)
plot(data(2,:),iscores(2,:),'o')
xlabel('Data'), ylabel('IC2 scores')
axis square

%% try again with mixed data


clear data

% two non-Gaussian distributions
dataO(1,:) = rand(1,N);
dataO(2,:) = rand(1,N).^4;

% then mix them into a new dataset
data(1,:) = .3*dataO(1,:) + .7*dataO(2,:);
data(2,:) = .7*dataO(1,:) + .3*dataO(2,:);


% ICA and scores (using the mixed data!)
b = jader(data);
iscores = b*data;


% plot distributions
figure(4), clf
% original data 1
subplot(321), 
hist(dataO(1,:),100)
title('O.D. 1')

% original data 2
subplot(322), 
hist(dataO(2,:),100)
title('O.D. 2')

% mixed data 1
subplot(323), 
hist(data(1,:),100)
title('Mixed data 1')

% mixed data 2
subplot(324), 
hist(data(2,:),100)
title('Mixed data 2')


% IC 1
subplot(325), 
hist(iscores(1,:),100)
title('IC 1')

% IC 2
subplot(326), 
hist(iscores(2,:),100)
title('IC 2')



% now plot data as a function of ICs
figure(5), clf
subplot(121)
plot(data(1,:),iscores(1,:),'o')
xlabel('Data'), ylabel('IC1 scores')
axis square

subplot(122)
plot(data(2,:),iscores(2,:),'o')
xlabel('Data'), ylabel('IC2 scores')
axis square


%% Distribution shapes of non-stationary sine waves

%%% The purpose of this cell is for you to see the signal value distributions 
%   of sine waves with varying degrees of non-stationarities.
%   Note: The code is hard-coded to five sine waves.

% some parameters
pnts  = 6000;
srate = 1000;

% frequencies
hz = linspace(0,srate,pnts);

% specify a range of FWHM
minFWHM = .1; % in Hz
maxFWHM = 10; % also in Hz.

% specify range of FWHM
fwhm = logspace(log10(.1),log10(10),5);
peakfreq = 8;

figure(6), clf

for fwhmi=1:5
    
    % create frequency-domain Gaussian
    s  = fwhm(fwhmi)*(2*pi-1)/(4*pi); % normalized width
    x  = hz-peakfreq;          % shifted frequencies
    fg = exp(-.5*(x/s).^2);    % gaussian
    
    % random Fourier coefficients and taper with the Gaussian
    fc = rand(1,pnts) .* exp(1i*2*pi*rand(1,pnts));
    fc = fc .* fg;
    
    % create the time-domain signal, and normalize
    sig = real( ifft(fc) );
    sig = (sig-mean(sig)) / std(sig);
    
    
    %%% plot the power spectrum
    subplot(5,3,fwhmi*3-2)
    plot(hz,abs(fc).^2)
    set(gca,'xlim',[0 peakfreq*4],'xtick',[],'ytick',[]), box off
##    title([ 'FWHM = ' num2str(round(fwhm(fwhmi),2)) ])
title([ 'FWHM = ' num2str(round(fwhm(fwhmi))) ])
    if fwhmi==5, xlabel('Frequency (Hz)'), end
    
    %%% plot the time-domain signal
    subplot(5,3,fwhmi*3-1)
    plot(sig), box off
    set(gca,'xtick',[],'ytick',[])
    if fwhmi==5, xlabel('Time'), end
    
    %%% plot the histogram
    subplot(5,3,fwhmi*3)
    [yh,xh]=hist(sig,90);
    plot(xh,yh)
    set(gca,'xlim',[-1 1]*5,'xtick',[],'ytick',[]), box off
    if fwhmi==5, xlabel('Data value'), end
end


%% done.

