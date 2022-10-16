function empVals = RESSfilterFGxTest(N,srate,f,fwhm,varargin)
% RESSfilterFGxTest   Test for spectral and temporal kernels in RESS method.
%  empVals = RESSfilterFGxTest(N,srate,f,fwhm,varargin)
% 
% 
%    INPUTS
%          N : number of time points in the to-be-filtered signal
%      srate : sampling rate in Hz
%          f : vector of 3 peak frequencies
%       fhwm : vector of 3 standard deviations
%              defined as full-width at half-maximum in Hz
%     fignum : (optional) figure number (useful for comparing different
%              parameter sets)
% 
%    OUTPUTS
%    empVals : vector of the empirical frequencies and FWHM (in Hz and in ms)
%              rows for frequencies, columns for parameters
% 
% Empirical frequency and FWHM depend on the sampling rate and the
% number of time points, and may thus be slightly different from
% the requested values.
% 
%  EXAMPLE:
%     empvals = RESSfilterFGxTest(4000,500,[10 11 12],[1 .5 1.5]);
% 
%  This script accompanies the method described in:
%    Cohen, Gulbinaite (2017). Rhythmic entrainment source separation: 
%      Optimizing analyses of neural responses to rhythmic sensory stimulation.
%      NeuroImage
% 
% 
% mikexcohen@gmail.com

%% input check

if numel(N)>1
    help RESSfilterFGxTest
    error('First input must be one number (N time points)')
end

if numel(f)~=3 || numel(fwhm)~=3
    help RESSfilterFGxTest
    error('Third and fourth inputs be be 3-element vectors')
end

if isempty(varargin)
    fignum = 20;
else
    fignum = varargin{1};
end


%% compute filters

% frequencies
hz = linspace(0,srate,N);

% initialize
[legu,legl,fx] = deal( cell(3,1) );
empVals = zeros(3);

% figure setups
figure(fignum),clf
linecols = 'krk';
tx = (-N/2:N/2-~mod(N,2))/srate;

for filti=1:3
    
    % create Gaussian
    s  = fwhm(filti)*(2*pi-1)/(4*pi); % normalized width
    x  = hz-f(filti);                 % shifted frequencies
    fx{filti} = exp(-.5*(x/s).^2);    % gaussian
    
    
    %% compute empirical frequencies and standard deviations
    
    idx = dsearchn(hz',f(filti));
    empVals(filti,1) = hz(idx);
    
    % find values closest to .5 after MINUS before the peak
    empVals(filti,2) = hz(idx-1+dsearchn(fx{filti}(idx:end)',.5)) - hz(dsearchn(fx{filti}(1:idx)',.5));
    
    % also temporal FWHM
    tmp = abs(hilbert(real(fftshift(ifft(fx{filti})))));
    tmp = tmp./max(tmp);
    [~,idxt] = max(tmp);
    empVals(filti,3) = (tx(idxt-1+dsearchn(tmp(idxt:end)',.5)) - tx(dsearchn(tmp(1:idxt)',.5)))*1000;
    
    %% show the plots
    
    % spectral shapes
    subplot(211), hold on
    hu(filti) = plot(hz,fx{filti},linecols(filti),'linew',2,'markerfacecolor','w');
    plot([hz(dsearchn(fx{filti}(1:idx)',.5)) hz(idx-1+dsearchn(fx{filti}(idx:end)',.5))],[fx{filti}(dsearchn(fx{filti}(1:idx)',.5)) fx{filti}(idx-1+dsearchn(fx{filti}(idx:end)',.5))],linecols(filti))
    legu{filti} = [ 'Requested: ' num2str(f(filti)) ', ' num2str(fwhm(filti)) ' Hz; Empirical: ' num2str(round(empVals(filti,1)*1000)/1000) ', ' num2str(round(empVals(filti,2)*1000)/1000) ' Hz' ];
    
    % temporal smoothing envelope
    subplot(212), hold on
    tmp1 = real(fftshift(ifft(fx{filti})));
    tmp1 = tmp1./max(tmp1);
    tmp2 = abs(hilbert(tmp1));
    plot(tx,tmp1,linecols(filti),'linew',.5);
    hl(filti) = plot(tx,tmp2,linecols(filti),'linew',2);
    legl{filti} = [ 'Empirical FWHM: ' num2str(empVals(filti,3)) ' ms' ];
    
end

%% make the plots look a bit nicer

subplot(211)
legend(hu,legu)
set(gca,'xlim',[max(f(1)-10,0) f(end)+10],'ylim',[0 1.01]);
xlabel('Frequency (Hz)'), ylabel('Amplitude gain')
title('Spectral shapes')

subplot(212)
legend(hl,legl)
xlabel('Time (s)'), ylabel('Amplitude gain')
set(gca,'xlim',([-1 1]*max(empVals(:,3))*1.1)/srate,'ylim',[-1.1 1.1]);
title('Temporal smoothing envelope')

%% done.
