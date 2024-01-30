
%%%%%%%%%%%%%%%%%%%
% jointfsrls.m
%%%%%%%%%%%%%%%%%%%

% Initialization

f(1)=0;            % phase
a=zeros(1,N);     % feedforward equalizer taps
b=zeros(1,M);     % feedback equalizer taps
c=[a -b];
x=zeros(1,N);      % input signal vector (in ff section)
Sf=0;              % PLL integrator output 
dd=zeros(1,M);     % vector of previous decisions ( in fb section)
sse=0;             % sum of squared errors        
%C=zeros(Nd,N+M);   % matrix of equalizer taps in time
P=eye(N+M)/delta; % inverse correlation matrix for RLS

%%%%%%%%%%%%%%%%%%
for n=1:Nd; 

%%%%read new sample%%%%

    nb=(n-1)*Ns+(Nplus-1)*Ns; % current input sample 
 %   xn=[];for l=1:FS; xn=[v(nb+ceil((2*l-1)*Ns/2/FS)) xn];end; % FS new input samples 
    xn=v(nb+ceil(Ns/FS/2):Ns/FS:nb+Ns); 
    x=[xn x];x=x(1:N); 

%%%%compute new signals%%%

    p=x*a'*exp(-j*f(n)); % output of ff section with corrected phase
    z(n)=p;
    q=dd*b'; % output of fb section
    de(n)=p-q; % estimate of data d(n)
    if n>Nt;  % Nt=training length;
%       d(n)=sign(real(de(n))); % BPSK
       d(n)=dec4psk(de(n)); % QPSK
%       d(n)=decision8(de(n));     % 8-QAM
%       d(n)=decision8psk(de(n));     % 8-PSK

    end; 
    e=d(n)-de(n); % error
    et(n)=abs(e.^2);
    sse=sse+abs(e^2);
    mse(n)=sse/n; mse(n);

%%%%parameter update%%%%

    y=[x*exp(-j*f(n)) dd]; % data vector for RLS
  
    Sf=Lf*Sf+imag(p*conj(d(n)+q));                 % PLL
    f(n+1)=f(n)+Kf1*imag(p*conj(d(n)+q))+Kf2*Sf;

    k=P/L*conj(y')/(1+conj(y)*P/L*conj(y'));       % RLS    
    c=c+conj(k')*conj(e);
    P=P/L-k*conj(y)*P/L;

    a=c(1:N);b=-c(N+1:N+M);
    dd=[d(n) dd]; dd=dd(1:M);
%    C(n,:)=[a -b];

    
end;
    
    
