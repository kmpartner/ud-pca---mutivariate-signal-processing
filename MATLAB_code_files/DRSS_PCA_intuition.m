%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Dimension reduction with PCA
%      VIDEO: PCA intuition with scatter plots and covariance surfaces
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%% PCA on simulated data

% data
x = [ 1*randn(1000,1) .4*randn(1000,1) ];

% rotation matrix
th = pi/4;
R1 = [ cos(th) -sin(th); sin(th) cos(th) ];

% rotate data to induce correlation
y = x*R1;


% plot the data in a scatter plot
figure(1), clf
subplot(131)
plot(y(:,1),y(:,2),'m.','markersize',17)

% make the plot look nicer
set(gca,'xlim',[-1 1]*max(y(:)),'ylim',[-1 1]*max(y(:)))
xlabel('x-axis'), ylabel('y-axis')
axis square

%% PCA via eigendecomposition

% mean-center
y = y - mean(y,1);

% covariance matrix
covmat = y'*y / (length(y)-1);

% eigendecomposition
[evecs,evals] = eig(covmat);


% plot the eigenvectors on the data
hold on
plot([0 evecs(1,1)]*evals(1),[0 evecs(2,1)]*evals(1),'k','linew',3)
plot([0 evecs(1,2)]*evals(end),[0 evecs(2,2)]*evals(end),'k','linew',3)


% show the covariance matrix
subplot(132)
imagesc(covmat), axis square
set(gca,'clim',[-1 1],'xtick',1:2,'ytick',1:2)
xlabel('Channels'), ylabel('Channels')

%% compute quadratic form

% weights along each dimension
xi = -2:.1:2;

quadform = zeros(length(xi));
for i=1:length(xi)
    for j=1:length(xi)
        
        % define vector
        x = [xi(i) xi(j)]';
        
        % QF
        quadform(i,j) = x'*covmat*x / (x'*x);
    end
end


% fill in missing point with 0
quadform(~isfinite(quadform)) = 0;

%% visualize the quadratic form surface

subplot(133)
surf(xi,xi,quadform'), shading interp
xlabel('W1'), ylabel('W2'), zlabel('energy')
rotate3d on, axis square

% and the eigenvectors
hold on
plot3([0 evecs(1,1)]*2,[0 evecs(2,1)]*2,[1 1]*max(quadform(:)),'w','linew',3)
plot3([0 evecs(1,2)]*2,[0 evecs(2,2)]*2,[1 1]*max(quadform(:)),'m','linew',3)

%%
