% Pulse shaping at the transmitter 

% d : input data sequence
% p : impulse response of the shaping filter 
% Ns : number of samples per symbol
% u : output signal

function u=fil(d,p,Ns);

N=length(d);
Lp=length(p);
Ld=Ns*N;

u=zeros(1,Lp+Ld-Ns);
for n=1:N;
    window=(n-1)*Ns+1:(n-1)*Ns+Lp;
    u(window)=u(window)+d(n)*p;
end;


