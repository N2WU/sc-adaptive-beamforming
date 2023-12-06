%% Define Variables
fc = 6.5e3;
duration = 1;
n_repeat = 4;
Fs = 96e3;
R = 3000;
Ns = 4;
channels = 12;
d0 = 0.05;
c = 343;
n_path = 1;
n_sim = 1;
reflection_list = [1 0.5];
x_tx_list = 5;
y_tx_list = 5;
SNR_db = 10;
SNR = 10^(SNR_db/10);

%% Generate BPSK Signal
% Generate gold code preamble
goldseq = comm.GoldSequence( ...
    "FirstPolynomial","x^3+x^2+1", ...
    "SecondPolynomial","x^3+x+1", ...
    "FirstInitialConditions",[0 0 1], ...
    "SecondInitialConditions",[0 1 1], ...
    "Index",3, ...
    "SamplesPerFrame",7);
outseq = goldseq()';
release(goldseq);

rc_tx = rcosdesign(0.5,R,Ns,"normal");

preambles = -1 + 2*repmat(outseq,[1 n_repeat]);
% upsample
preamble_rc = conv(preambles,rc_tx);
preamble_rc = preamble_rc(1:12000); %maybe this is why it upsamples
preamble_xcorr = conv(-1+2*outseq,rc_tx);

s = real(preamble_rc.*exp(2*pi*1i*fc*(1:length(preamble_rc))/Fs));

%% Simulate Channel

r_multichannel = rand(channels,duration*Fs)/SNR;
% maybe no delays, etc for now
r_multichannel = r_multichannel + resample(s,Fs*duration,length(s));

%% Receive Signal
v = r_multichannel*exp(-2*pi*1i*fc*(1:length(r_multichannel(1,:)))'/Fs);
% decimate
v = conv(v,rc_tx);

%% Process Data
% Cross-correlation
figure
plot(abs(xcorr(v,preambles)))
% Constellation Diagram
figure
scatter(preambles,zeros(1,length(preambles)),"o")
hold on
scatter(v,zeros(1,length(v)),"x")
hold off
% Channel Response
% this should show the delays after a while