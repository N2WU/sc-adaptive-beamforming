%% generate v
clear, clc
close all

passband=1;
Nd=3000;
Nz=100;

dp=[1 -1 1 -1 1 1 -1 -1 1 1 1 1 1]*(1+1i)/sqrt(2);
fc=17000;
Fs=44100; fs=Fs/4; Ts=1/fs;
alpha=0.25;trunc=4;
Ns=7; T=Ns*Ts; R=1/T;

g=rcos(alpha,Ns,trunc);
up=fil(dp,g,Ns);
lenu=length(up);

d=sign(randn(1,Nd))+1i*sign(randn(1,Nd));d=d/sqrt(2);
ud=fil(d,g,Ns);
u=[up zeros(1,Nz*Ns) ud];
us=resample(u,Fs,fs);
%s=real(us.*exp(1i*2*pi*fc*(0:length(us)-1)/Fs));

K0 = 10;
Ns = 7;
Nplus = 4;
vk = [];

for k = 1:K0    % repeat v 10 K0 times, add noise to each signal
    Tmp=0;
    
    snr=300; SNR=10^(snr/10);

    v=zeros(1,length(u));

    vp=u; 
    vp(length(u))=0;
    v=v+vp;

    v=v/sqrt(pwr(v)); 
    v0 = v;
    v = v0;
    z=sqrt(1/(2*SNR))*randn(size(v))+1i*sqrt(1/(2*SNR))*randn(size(v));
    %z = ones(size(v)) + 1i*ones(size(v));
    % hmmm
    v=v+z-z;
    
    lenv=length(v);

    v=v(lenu+Nz*Ns+trunc*Ns+1+1:end);
    
    v = resample(v,2,Ns);
    v = [v zeros(1,Nplus*2)];
    vk(k,:) = v;
end

%%
%vk_real = readNPY('C:\Users\Nolan\Documents\GitHub\sc-adaptive-beamforming\data\vk_real.npy');
%vk_imag = readNPY('C:\Users\Nolan\Documents\GitHub\sc-adaptive-beamforming\data\vk_imag.npy');
%vk = vk_real + 1j*vk_imag;
%writeNPY(real(vk),'C:\Users\Nolan\Documents\GitHub\sc-adaptive-beamforming\data\vk_real.npy')
%writeNPY(imag(vk),'C:\Users\Nolan\Documents\GitHub\sc-adaptive-beamforming\data\vk_imag.npy')

%d_real = readNPY('C:\Users\Nolan\Documents\GitHub\sc-adaptive-beamforming\data\d_real.npy');
%d_imag = readNPY('C:\Users\Nolan\Documents\GitHub\sc-adaptive-beamforming\data\d_imag.npy');
%d = d_real + 1j*d_imag;
%writeNPY(real(d),'C:\Users\Nolan\Documents\GitHub\sc-adaptive-beamforming\data\d.npy')
%writeNPY(imag(d),'C:\Users\Nolan\Documents\GitHub\sc-adaptive-beamforming\data\d.npy')

figure
K = length(vk(:,1));
Ns = 2;
Tmp = 0;
N=6*Ns; M=ceil(Tmp/T); delta=10^(-3); Nt=4*(N+M); FS=2;
Kf1=0.001; Kf2=Kf1/10; Lf=1; L=0.98;
P = eye(K*N+M)/delta;
Lbf=0.99;

v = vk(1:K,:);

%

f = zeros(Nd,K);

a = zeros(1,K*N);
b = zeros(1,M);
c = [a -b];
p = zeros(1,K);
d_tilde = zeros(1,M);
Sf = zeros(1,K);    % sum of phi_0 ~ phi_n
x = zeros(K,N);
et = zeros(Nd,1);

for n = 1:Nd
    nb = (n-1) * Ns + (Nplus-1) * Ns;
    xn = v(:, nb + ceil(Ns/FS/2) : Ns/FS : nb + Ns);
    for k = 1:K
        xn(k,:) = xn(k,:)*exp(-1i*f(n,k));
    end
    xn = fliplr(xn);
    x = [xn x];
    x = x(:, 1:N);

    for k = 1:K
        p(k) = x(k,:)*a(1,(k-1)*N+1:k*N)';
    end
    psum = sum(p);

    q = d_tilde*b';
    d_hat(n) = psum-q;

    if n > Nt
        d(n) = dec4psk(d_hat(n)); % make decision
    end
    
    e = d(n) - d_hat(n);
    et(n) = abs(e.^2);

    % parameter update
    phi = imag(p.*conj(p+e));
    Sf = Lf*Sf + phi;
    f(n+1,:) = f(n,:) + Kf1*phi + Kf2*Sf;

    y = reshape(x.', 1, K*N);
    y = [y d_tilde];

    k = P/L*y.' / (1+conj(y)*P/L*y.');
    c = c + k.'*conj(e);
    P = P/L - k*conj(y)*P/L;

    a = c(1:K*N); b = -c(K*N+1:K*N+M);
    d_tilde = [d(n) d_tilde]; d_tilde = d_tilde(1:M);
end


% plot
snr=300; SNR=10^(snr/10);
plot(d_hat(Nt:end), '*');
axis('square')
axis([-2 2 -2 2]);
title(['K=',num2str(K),' SNR=',num2str(snr),'dB'])