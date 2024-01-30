

subplot(221)
plot(10*log10(et))
axis('square')
title('squared error')
xlabel('[symbol intervals]')


subplot(222)

plot(f);
axis('square')
axis([0 length(de) min(min(f))-0.1 max(max(f))+0.1]);
%text(100,min(min(ff)-0.05),['fd=',num2str(mean(fd)),'Hz'],'sc')
%plot(f)
title('phase [rad]')
xlabel('[symbol intervals]')

subplot(223)

plot(de(Nt:length(de)),'*');
axis('square')
axis([-2 2 -2 2])
%axis([-2 2 -2 2]);
title('out.scatter');
xlabel('Re')
ylabel('Im')



subplot(224)
axis('off')

text(0,0.9,['receiver parameters (',int2str(R),' sps)'],'sc')
text(0,0.7,['N=',num2str(N), ' (T/',  num2str(FS), ')', '  M=',num2str(M)],'sc')
text(0,0.5,['Kf1=',num2str(Kf1),' Kf2=',num2str(Kf1/10)],'sc')
% if K>Ke;
% text(0,0.6,['K=',num2str(K),' P=',num2str(Ke)],'sc')
% text(0,0.5,['Le=',num2str(L),' Lc=',num2str(Lbf),],'sc')
% end;
% if K==Ke;
% text(0,0.6,['K=',num2str(K)],'sc')
 text(0,0.6,['\lambda =',num2str(L)],'sc')
% end;
text(0,0.4,['Nt=',num2str(Nt)],'sc')

text(0,0.2,['Pe~',num2str(Ner/Nd)],'sc')
%text(0,0.0,['SNRout~',num2str(snr),'dB'],'sc')







