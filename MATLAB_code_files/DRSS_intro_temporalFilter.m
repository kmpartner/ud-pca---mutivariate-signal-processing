%%
%     COURSE: Dimension reduction and source separation in neuroscience
%    SECTION: Introduction
%      VIDEO: What are linear spatial filters?
% Instructor: mikexcohen.com
%
%%

% number of time points
n = 1000;

% time vector
t = linspace(-2,2,n);

% create signal
s = zeros(n,1);
s(n/2-100:n/2+100) = 1;
% s = smooth(s+randn(size(s))/20,5);
s = s+randn(size(s))/20;

% create filter kernel
w = cos(2*pi*10*t) .* exp( -t.^2/.01 );


%% plotting

% signal
subplot(311)
plot(t,s,'k','linew',3)

% kernel
subplot(312)
plot(t,w,'k')

% result of filtering
subplot(313)
plot(t,conv(s,w,'same'),'k','linew',3)

%%

