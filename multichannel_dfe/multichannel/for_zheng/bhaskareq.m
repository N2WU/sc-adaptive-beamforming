%bhaskareq

clear all;

passband=1;
Nd=3000;
Nz=100;


dp=[1 -1 1 -1 1 1 -1 -1 1 1 1 1 1]*(1+j)/sqrt(2);
fc=17000;
Fs=44100; fs=Fs/4; Ts=1/fs;
alpha=0.25;trunc=4;
Ns=7; T=Ns*Ts; R=1/T; B=R*(1+alpha);
Nso=Ns;

%fmin=(fc-B/2)/1000, fmax=(fc+B/2)/1000, R/1000

g=rcos(alpha,Ns,trunc);
up=fil(dp,g,Ns);
lenu=length(up);

d=sign(randn(1,Nd))+j*sign(randn(1,Nd));d=d/sqrt(2);
ud=fil(d,g,Ns);
u=[up zeros(1,Nz*Ns) ud];
us=resample(u,Fs,fs);
s=real(us.*exp(j*2*pi*fc*(0:length(us)-1)/Fs));

save bhaskar1 s d Fs


Tmp=40/1000;
tau=(3+[0 5 7 14 27 33])/1000;
h=exp(-tau/Tmp); h=h/sqrt(sum(abs(h)));
taus=tau/Ts; taus=round(taus);

snr=20; SNR=10^(snr/10);

v=1; c=350; a=v/c; fd=a*fc; 

lenv=length(u)+max(taus);
v=zeros(1,lenv);
for p=1:length(tau);
    taup=taus(p);
    vp=[zeros(1,taup) h(p)*u];vp(lenv)=0;
    v=v+vp;
end;
v=v/sqrt(pwr(v)); 
if passband==0; 
    vs=resample(v,Fs,fs);
    vr=resample(vs,10^4,round((1+a)*10^4));
    vr=vr.*exp(j*2*pi*fd*[0:length(vr)-1]/Fs);
    v=decimate(vr,Fs/fs); %vbase=v;
end;
if passband==1;
    vs=resample(v,Fs,fs);
    s=real(vs.*exp(j*2*pi*fc*(0:length(vs)-1)/Fs));
    r=resample(s,10^4,round((1+a)*10^4)); % r(t)=s((1+a)t);
    vr=2*r.*exp(-j*2*pi*fc*(0:length(r)-1)/Fs);
    v=decimate(vr,Fs/fs); %vpass=v;
end;
z=sqrt(1/(2*SNR))*randn(size(v))+j*sqrt(1/(2*SNR))*randn(size(v));
v=v+z;
lenv=length(v);
vin=v;
vp=v(1:length(up)+Nz*Ns);

[del,x]=fdel(vp,up);
tau0e=round(del*Ts*1000)
vp1=vp(del:del+lenu-1);

fde=fdop(vp1,up,fs,12);
fde, fde1=fde;
v=v.*exp(-j*2*pi*[0:lenv-1]*fde*Ts);
v=resample(v,10^4,round(1/(1+fde/fc)*10^4));

v=v(del:del+length(u)-1);
v=v(lenu+Nz*Ns+trunc*Ns+1+1:end);


% lenv=length(v);
% vp=v(1:lenu+Nz*Ns);
% 
% [del,x]=fdel(vp,up);
% round(del*Ts*1000)
% vp2=vp(del:del+lenu-1);
% v2=v(del:del+length(u)-1);
% 
% fde=fdop(vp2,up,fs,12);
% fde, fde2=fde;
% v=v2.*exp(-j*2*pi*[0:length(u)-1]*fde*Ts);
% v=v(lenu+Nz*Ns+trunc*Ns+1+1:end);
fde2=0;


figure(1)
stem(tau*1000,h)
xlabel('delay [ms]')
title('channel and system parameters')
text(0.25,0.9,['v = ',num2str(a*c),' m/s'],'sc')
text(0.55,0.85,['f_c = ',num2str(fc/1000),' kHz'],'sc')
text(0.55,0.8,['B = ',num2str(B/1000),' kHz'],'sc')
text(0.55,0.75,['R = ',num2str(R/1000),' ksps'],'sc')
text(0.55,0.7,['(roll-off ', num2str(alpha),')'],'sc')
text(0.75,0.85,['F_s = ',num2str(Fs/1000),' kHz'],'sc')
text(0.75,0.8,['f_s = F_s/4 = ', int2str(fs/R),' R'],'sc')
print -depsc bhaskar1

figure(2)
subplot(311)
plot((0:length(u)-1)*Ts*1000,abs(u))
xlabel('time [ms]')
ylabel('tx')
axis([0 (length(u)+round(Tmp*fs))*Ts*1000 0 max(abs(u))*1.1])
%text(0.6,0.75,['Barker (baseband), roll-off ', num2str(alpha)],'sc')
subplot(312)
vin=vin/max(abs(vin))*max(abs(u));
plot((0:length(vin)-1)*Ts*1000,abs(vin))
axis([0 (length(u)+round(Tmp*fs))*Ts*1000 0 max(abs(u))*1.1])
xlabel('time [ms]')
ylabel('rx')
%text(0.75,0.75,['SNR=', num2str(snr),' dB'],'sc')
subplot(313)
xwin=[1:length(vp)];
plot((xwin-1)*Ts*1000,abs(x(xwin)/max(abs(x(xwin)))*1))
xlabel('delay [ms]')
ylabel('sync')
text(0.75,0.85,['est. delay=', num2str(round(del*Ts*1000)),' ms'],'sc')
text(0.75,0.7,['est. Doppler=', num2str(round((fde1+fde2)/fc*10^3)/10^3)],'sc')
text(0.45,0.85,['true delay=', num2str(round(taus(1)*Ts*1000)),' ms'],'sc')
text(0.45,0.7,['true Doppler=', num2str(round(a*10^3)/10^3)],'sc')
text(0.75,0.5,['SNR=', num2str(snr),' dB'],'sc')
print -depsc bhaskar2

v=resample(v,2,Ns); Ns=2;
Nplus=4; N=6*Ns; M=ceil(Tmp/T); delta=10^(-3); Nt=4*(N+M); FS=2;
Kf1=0.001; Kf2=Kf1/10; Lf=1; L=0.98;
v=[v zeros(1,Nplus*Ns)];
dtrue=d;
jointfsrls;

err=d-dtrue;err=err(Nt:end);Ner=length(find(abs(err)>0.01))
figure(3); clf;
plotbhaskar;
print -depsc bhaskar3

[del,x]=fdel(v(1:2:end),dtrue);
%plot(abs(x));del
    
    

