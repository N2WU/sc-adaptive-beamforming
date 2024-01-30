function [fd,X,N]=fdop(v,u,fs,Ndes);

x=v.*conj(u);
N=ceil(log10(length(x))/log10(2)); % first 2^N that will do
if Ndes>N; N=Ndes; end; 
X=fft(x,2^N)/length(x);
X=fftshift(X);
f=((1:2^N)-2^N/2-1)/(2^N)*fs; % should have -1
% plot(f,abs(X))
% axis('square')
% xlabel('f [Hz]')
% ylabel('spectrum')
[m,i]=max(X);
fd=f(i);




