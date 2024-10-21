clear all
close all

setFigDef()

%% read result
[num, txt,raw] = xlsread('/Users/meidi/Desktop/rice/bennettlab/Modulus/manuscript/growth.xlsx');
data_idx = ~isnan(num(:, 1));
data = num(data_idx,[1,2,5,6,9,10]);

tw = data([1:72],:);
pf = data([73:153],:);
dif = data([154:234],:);
qs = data([235:315],:);


%% plot tw growth
% t1 = tw(4:end,1);
% y1 = tw(4:end,2);
% t2 = tw(4:21,3);
% y2 = tw(4:21,4);
% t3 = tw(4:18,5);
% y3 = tw(4:18,6);
% 
% times = linspace(t1(1),t1(end));
% fun = @(par,t) par(1)./(1+(par(1)-par(2))/par(2)*exp(-par(3)*t));
% par0 = [120000,1000,1];
% par1 = lsqcurvefit(fun,par0,t1,y1);
% 
% figure
% plot(t1,y1,'ko',times,fun(par1,times),'b-')
% 
% 
% par2 = lsqcurvefit(fun,par0,t2,y2);
% times2 = linspace(t2(1),t2(end));
% figure
% plot(t2,y2,'ko',times,fun(par2,times),'b-')
% 
% 
% par3 = lsqcurvefit(fun,par0,t3,y3);
% times3 = linspace(t3(1),t3(end));
% figure
% plot(t3,y3,'ko',times,fun(par3,times),'b-')

tw_sh1= reshape(tw(:,2),[9,8]);
tw_sh2= reshape(tw(:,4),[9,8]);
tw_sh3= reshape(tw(:,6),[9,8]);
tw_ave1 = mean(tw_sh1);
tw_ave2 = mean(tw_sh2);
tw_ave3 = mean(tw_sh3);
tw_std1 = std(tw_sh1);
tw_std2 = std(tw_sh2);
tw_std3 = std(tw_sh3);
tw_t1 = reshape(tw(:,1),[9,8]);
tw_t2 = reshape(tw(1:63,3),[9,7]);
tw_t3 = reshape(tw(1:54,5),[9,6]);

figure
hold on
errorbar(tw_t1(1,:),tw_ave1,tw_std1,'MarkerSize',20,'LineWidth',2,'color','#DC582A')
errorbar(tw_t2(1,:),tw_ave2(1:7),tw_std2(1:7),'MarkerSize',20,'LineWidth',2,'color','#ADD8E6')
errorbar(tw_t3(1,:),tw_ave3(1:6),tw_std3(1:6),'MarkerSize',20,'LineWidth',2,'color','#FFC000')
plot(tw(:,1),tw(:,2),'o','color','#DC582A')
plot(tw(:,3),tw(:,4),'o','color','#ADD8E6')
plot(tw(:,5),tw(:,6),'o','color','#FFC000')
legend('OD = 0.2','OD = 0.6','OD = 1.8','location','northwest')
xlabel('time [hr]')
ylabel('cell density [CFU]')
set(gca, 'YScale', 'log')
box on
title('QS+ PF+')

% figure
% hold on
% plot(tw(:,1),tw(:,2),'o','color','#DC582A')
% plot(tw(:,3),tw(:,4),'o','color','#ADD8E6')
% plot(tw(:,5),tw(:,6),'o','color','#FFC000')
% plot(times,fun(par1,times),'-','color','#DC582A')
% plot(times,fun(par2,times),'-','color','#ADD8E6')
% plot(times,fun(par3,times),'-','color','#FFC000')
% legend('OD = 0.2','OD = 0.6','OD = 1.8','location','northwest')
% xlabel('time(hr)')
% ylabel('cell density (CFU)')
% % set(gca, 'YScale', 'log')


%% plot pf growth

pf_sh1= reshape(pf(:,2),[9,9]);
pf_sh2= reshape(pf(:,4),[9,9]);
pf_sh3= reshape(pf(:,6),[9,9]);
pf_ave1 = mean(pf_sh1);
pf_ave2 = mean(pf_sh2);
pf_ave3 = mean(pf_sh3);
pf_std1 = std(pf_sh1);
pf_std2 = std(pf_sh2);
pf_std3 = std(pf_sh3);
pf_t1 = reshape(pf(:,1),[9,9]);
pf_t2 = reshape(pf(:,3),[9,9]);
pf_t3 = reshape(pf(:,5),[9,9]);

figure
hold on
errorbar(pf_t1(1,:),pf_ave1,pf_std1,'MarkerSize',20,'LineWidth',2,'color','#DC582A')
errorbar(pf_t2(1,:),pf_ave2,pf_std2,'MarkerSize',20,'LineWidth',2,'color','#ADD8E6')
errorbar(pf_t3(1,:),pf_ave3,pf_std3,'MarkerSize',20,'LineWidth',2,'color','#FFC000')
plot(pf(:,1),pf(:,2),'o','color','#DC582A')
plot(pf(:,3),pf(:,4),'o','color','#ADD8E6')
plot(pf(:,5),pf(:,6),'o','color','#FFC000')
legend('OD = 0.2','OD = 0.6','OD = 1.8','location','northwest')
xlabel('time [hr]')
ylabel('cell density [CFU]')
set(gca, 'YScale', 'log')
box on
title('QS- PF+')

% t1 = pf(10:end,1);
% y1 = pf(10:end,2);
% t2 = pf(10:end,3);
% y2 = pf(10:end,4);
% t3 = pf(10:end,5);
% y3 = pf(10:end,6);
% 
% times = linspace(0,t3(end));
% fun = @(par,t) par(1)./(1+(par(1)-par(2))/par(2)*exp(-par(3)*t));
% par0 = [120000,1000,1];
% par1 = lsqcurvefit(fun,par0,t1,y1);
% 
% figure
% plot(t1,y1,'ko',times,fun(par1,times),'b-')
% 
% par2 = lsqcurvefit(fun,par0,t2,y2);
% times2 = linspace(t2(1),t2(end));
% figure
% plot(t2,y2,'ko',times,fun(par2,times2),'b-')
% 
% 
% par3 = lsqcurvefit(fun,par0,t3,y3);
% times3 = linspace(t3(1),t3(end));
% figure
% plot(t3,y3,'ko',times,fun(par3,times3),'b-')
% 
% figure
% hold on
% plot(pf(:,1),pf(:,2),'o','color','#DC582A')
% plot(pf(:,3),pf(:,4),'o','color','#ADD8E6')
% plot(pf(:,5),pf(:,6),'o','color','#FFC000')
% plot(times3,fun(par1,times3),'-','color','#DC582A')
% plot(times3,fun(par2,times3),'-','color','#ADD8E6')
% plot(times3,fun(par3,times3),'-','color','#FFC000')
% legend('OD = 0.2','OD = 0.6','OD = 1.8','location','northwest')
% xlabel('time(hr)')
% ylabel('cell density (CFU)')
% set(gca, 'YScale', 'log')

%% QS- PF- circuit
dif_sh1= reshape(dif(:,2),[9,9]);
dif_sh2= reshape(dif(:,4),[9,9]);
dif_sh3= reshape(dif(:,6),[9,9]);
dif_ave1 = mean(dif_sh1);
dif_ave2 = mean(dif_sh2);
dif_ave3 = mean(dif_sh3);
dif_std1 = std(dif_sh1);
dif_std2 = std(dif_sh2);
dif_std3 = std(dif_sh3);
t = reshape(dif(:,1),[9,9]);


figure
hold on
errorbar(t(1,:),dif_ave1,dif_std1,'MarkerSize',20,'LineWidth',2,'color','#DC582A')
errorbar(t(1,:),dif_ave2,dif_std2,'MarkerSize',20,'LineWidth',2,'color','#ADD8E6')
errorbar(t(1,1:8),dif_ave3(1:8),dif_std3(1:8),'MarkerSize',20,'LineWidth',2,'color','#FFC000')
plot(dif(:,1),dif(:,2),'o','color','#DC582A')
plot(dif(:,3),dif(:,4),'o','color','#ADD8E6')
plot(dif(:,5),dif(:,6),'o','color','#FFC000')
legend('OD = 0.2','OD = 0.6','OD = 1.8','location','northwest')
xlabel('time [hr]')
ylabel('cell density [CFU]')
set(gca, 'YScale', 'log')
box on
title('QS- PF-')


%% QS+ PF-
qs_sh1= reshape(qs(:,2),[9,9]);
qs_sh2= reshape(qs(:,4),[9,9]);
qs_sh3= reshape(qs(:,6),[9,9]);
qs_ave1 = mean(qs_sh1);
qs_ave2 = mean(qs_sh2);
qs_ave2(1) = mean(qs_sh2([1:3,5,7:9],1));
qs_ave3 = mean(qs_sh3);
qs_std1 = std(qs_sh1);
qs_std2 = std(qs_sh2);
qs_std3 = std(qs_sh3);
t = reshape(qs(:,1),[9,9]);


figure
hold on
errorbar(t(1,:),qs_ave1,qs_std1,'MarkerSize',20,'LineWidth',2,'color','#DC582A')
errorbar(t(1,:),qs_ave2,qs_std2,'MarkerSize',20,'LineWidth',2,'color','#ADD8E6')
errorbar(t(1,:),qs_ave3,qs_std3,'MarkerSize',20,'LineWidth',2,'color','#FFC000')
plot(qs(:,1),qs(:,2),'o','color','#DC582A')
plot(qs(:,3),qs(:,4),'o','color','#ADD8E6')
plot(qs(:,5),qs(:,6),'o','color','#FFC000')
legend('OD = 0.2','OD = 0.6','OD = 1.8','location','northwest')
xlabel('time [hr]')
ylabel('cell density [CFU]')
set(gca, 'YScale', 'log')
box on
title('QS+ PF-')