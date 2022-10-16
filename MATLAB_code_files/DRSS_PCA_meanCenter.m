%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Dimension reduction with PCA
%      VIDEO: MATLAB: importance of mean-centering for PCA
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%%

% data
x = [ randn(1000,1) .2*randn(1000,1) ];

% rotation matrix
th = -pi/4;
R1 = [ cos(th) -sin(th); sin(th) cos(th) ];

% rotate data to induce correlation
y = x*R1;



%%% add a mean offset %%%
y(:,1) = y(:,1) + 0;
##y(:,1) = y(:,1) + 1;


% PCA of y (correlated data)
covmat = y'*y / (length(y)-1);
[evecs,evals] = eig(covmat);

% (optional) mean-center the data for display
## y = y - mean(y);



% plot the data
figure(1), clf

% plot the data in a scatter plot
plot(y(:,1),y(:,2),'m.','markersize',17)

% plot the eigenvectors
hold on
% plot([0 evecs(1,1)],[0 evecs(2,1)],'k','linew',4)
% plot([0 evecs(1,2)],[0 evecs(2,2)],'k','linew',4)
plot([0 evecs(1,1)]*evals(1),[0 evecs(2,1)]*evals(1),'k','linew',4)
plot([0 evecs(1,2)]*evals(end),[0 evecs(2,2)]*evals(end),'k','linew',4)

% make the plot look nicer
set(gca,'xlim',[-1 1]*max(y(:)),'ylim',[-1 1]*max(y(:)))
xlabel('x-axis'), ylabel('y-axis')
axis square


%%
