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
Ns=7; T=Ns*Ts; R=1/T; B=R*(1+alpha);
Nso=Ns;

%fmin=(fc-B/2)/1000, fmax=(fc+B/2)/1000, R/1000

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

load = 1;

for k = 1:K0    % repeat v 10 K0 times, add noise to each signal
    Tmp=40/1000;
    % tau=(3+[0 5 7 14 27 33])/1000;
    tau = (3+randi(33,1,6))/1000;
    h=exp(-tau/Tmp); h=h/sqrt(sum(abs(h)));
    taus=tau/Ts; taus=round(taus);
    
    snr=15; SNR=10^(snr/10);
    
    v=1; c=350; a=v/c; fd=a*fc; 
    
    lenv=length(u)+max(taus);
    v=zeros(1,lenv);
    for p=1:length(tau)
        taup=taus(p);
        vp=[zeros(1,taup) h(p)*u];vp(lenv)=0;
        v=v+vp;
    end
    v=v/sqrt(pwr(v)); 
    if passband==0 
        vs=resample(v,Fs,fs);
        vr=resample(vs,10^4,round((1+a)*10^4));
        vr=vr.*exp(1i*2*pi*fd*[0:length(vr)-1]/Fs);
        v=decimate(vr,Fs/fs); %vbase=v;
    end
    if passband==1
        vs=resample(v,Fs,fs);
        s=real(vs.*exp(1i*2*pi*fc*(0:length(vs)-1)/Fs));
        r=resample(s,10^4,round((1+a)*10^4)); % r(t)=s((1+a)t);
        z=sqrt(1/(2*SNR))*randn(size(r))+1i*sqrt(1/(2*SNR))*randn(size(r));
        r = r + z;
        vr=2*r.*exp(-1i*2*pi*fc*(0:length(r)-1)/Fs);
        v=decimate(vr,Fs/fs); %vpass=v;
    end

    v0 = v;
    v = v0;
    %z=sqrt(1/(2*SNR))*randn(size(v))+1i*sqrt(1/(2*SNR))*randn(size(v));
    %v=v+z;
    lenv=length(v);
    vin=v;
    vp=v(1:length(up)+Nz*Ns);
    
    [del,~]=fdel(vp,up);
    %tau0e=round(del*Ts*1000);
    vp1=vp(del:del+lenu-1);
    
    fde=fdop(vp1,up,fs,12);
    %fde1=fde;
    v=v.*exp(-1i*2*pi*[0:lenv-1]*fde*Ts);
    v=resample(v,10^4,round(1/(1+fde/fc)*10^4));
    
    v=v(del:del+length(u)-1);
    v=v(lenu+Nz*Ns+trunc*Ns+1+1:end);
    v = resample(v,2,Ns);
    v = [v zeros(1,Nplus*2)];
    vk(k,:) = v;
end

%%

if load == 1
    vk_real = readNPY('../../data/vk_real.npy');
    vk_imag = readNPY('../../data/vk_imag.npy');
    vk = vk_real + 1j*vk_imag;
    d_real = readNPY('../../data/d_real.npy');
    d_imag = readNPY('../../data/d_imag.npy');
    d = d_real + 1j*d_imag;
else
    writeNPY(real(vk),'../../data/vk_real.npy')
    writeNPY(imag(vk),'../../data/vk_imag.npy')
    writeNPY(real(d),'../../data/d_real.npy')
    writeNPY(imag(d),'../../data/d_imag.npy')
end

figure;
K = length(vk(:,1));

Ns = 2;
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

mse = 10*log10(mean(abs(d(1+Nt:end)-d_hat(Nt+1:end).').^2))

% plot

plot(d_hat(Nt:end), '*');
axis('square')
axis([-2 2 -2 2]);
title(['K=',num2str(K),' SNR=',num2str(snr),'dB'])

% subplot(122)
% plot(10*log10(et))
% axis('square')
