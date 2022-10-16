%%
%     COURSE: PCA and multivariate neural signal processing
%    SECTION: Source separation with GED
%      VIDEO: ZCA demo
% Instructor: sincxpress.com
%
%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear, clc

%% 

d = linspace(.2,2,14);

figure(1), clf
plot(d,d.^(-1/2),'ks-','linew',2,'markerfacecolor','w','markersize',13)

hold on
plot([d(1) 1],[1 1],'k--')
yy = get(gca,'ylim');
plot([1 1],[yy(1) 1],'k--')

axis square
xlabel('d')
ylabel('d^{-1/2}')


%% generate 2D data

% number of data points
N = 1000;


% part 1 of the data
x1 = [ 1*randn(N/2,1) .1*randn(N/2,1) ];
% rotation matrix
th = pi/4;
R1 = [ cos(th) -sin(th) ;
       sin(th)  cos(th) ];


% part 2 of the data
x2 = [ 1*randn(N/2,1) .1*randn(N/2,1) ];
% note the different rotation angle
th = pi;
R2 = [ cos(th) -sin(th) ;
       sin(th)  cos(th) ];

   
   
% combine and rotate data
y = [x1*R1; x2*R2];


%% eigendecomposition for PCA and ZCA

% eig of y
y = bsxfun(@minus,y,mean(y,1));
covmat = y'*y / (length(y)-1);
[V,D] = eig(covmat);

% component data
c = y*V;



% ZCA
yz = ( V*D^(-1/2)*V'*y' )';

% now PCA of ZCA data (question: Do you need the N-1 scaling?)
[Vz,Dz] = eig( yz'*yz );
cz = yz*Vz;

%% plotting

figure(2), clf

% original data
subplot(221), hold on
plot(y(:,1),y(:,2),'ko','markerfacecolor','m')
plot([0 V(1,1)],[0 V(2,1)],'k','linew',3)
plot([0 V(1,2)],[0 V(2,2)],'k','linew',3)
axis square
axis([-1 1 -1 1]*max(abs(y(:))))
xlabel('y_1'), ylabel('y_2')
title('Data')

% component projections
subplot(222)
plot(c(:,1),c(:,2),'ko','markerfacecolor','m')
axis square
axis([-1 1 -1 1]*max(abs(y(:))))
xlabel('pc_1'), ylabel('pc_2')
title('PCA of data')


% whitened data
subplot(223), hold on
plot(yz(:,1),yz(:,2),'ko','markerfacecolor','m')
plot(yz(1,1),yz(1,2),'rs','markerfacecolor','y','markersize',15)
axis square
axis([-1 1 -1 1]*max(abs(yz(:))))
xlabel('y_{z1}'), ylabel('y_{z2}')
title('Whitened')
plot([0 Vz(1,1)],[0 Vz(2,1)],'k','linew',3)
plot([0 Vz(1,2)],[0 Vz(2,2)],'k','linew',3)


subplot(224), hold on
plot(cz(:,1),cz(:,2),'ko','markerfacecolor','m')
plot(cz(1,1),cz(1,2),'rs','markerfacecolor','y','markersize',15)
axis square
axis([-1 1 -1 1]*max(abs(yz(:))))
xlabel('pc_1'), ylabel('pc_2')
title('PCA of whitened')

%%
