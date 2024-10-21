clear all
close all

setFigDef()

crt_path = cd();
namepath1 = [crt_path '/experiment data/IS-PF-/'];
namepath2 = [crt_path '/experiment data/IS+PF-/'];
namepath3 = [crt_path '/experiment data/IS-PF+/'];
namepath4 = [crt_path '/experiment data/IS+PF+/'];
%% read agar plate result
namepath =[crt_path '/experiment data/leaky fluorescence/'];

[yfp_503_1,~] = imread(strcat(namepath,'503-1_YFP.tif'));
[yfp_503_2,~] = imread(strcat(namepath,'503-2_YFP.tif'));
[yfp_503_3,~] = imread(strcat(namepath,'503-3_YFP.tif'));
[yfp_502_1,~] = imread(strcat(namepath,'502-1_YFP.tif'));
[yfp_502_2,~] = imread(strcat(namepath,'502-2_YFP.tif'));
[yfp_502_3,~] = imread(strcat(namepath,'502-3_YFP.tif'));
[yfp_146_1,~] = imread(strcat(namepath,'146-1_YFP_20231101.tif'));
[yfp_146_2,~] = imread(strcat(namepath,'146-2_YFP_20231101.tif'));
[yfp_146_3,~] = imread(strcat(namepath,'146-3_YFP_20231101.tif'));
[yfp_476_1,~] = imread(strcat(namepath,'476-1_YFP_20231101.tif'));
[yfp_476_2,~] = imread(strcat(namepath,'476-2_YFP_20231101.tif'));
[yfp_476_3,~] = imread(strcat(namepath,'476-3_YFP_20231101.tif'));
[yfp_678_1,~] = imread(strcat(namepath,'678-1_YFP.tif'));
[yfp_678_2,~] = imread(strcat(namepath,'678-2_YFP.tif'));
[yfp_678_3,~] = imread(strcat(namepath,'678-3_YFP.tif'));
[yfp_628_1,~] = imread(strcat(namepath,'628-1_YFP.tif'));
[yfp_628_2,~] = imread(strcat(namepath,'628-2_YFP.tif'));
[yfp_628_3,~] = imread(strcat(namepath,'628-3_YFP.tif'));
[yfp_677_1,~] = imread(strcat(namepath,'677-1_YFP_20231025.tif'));
[yfp_677_2,~] = imread(strcat(namepath,'677-2_YFP_20231025.tif'));
[yfp_677_3,~] = imread(strcat(namepath,'677-3_YFP_20231025.tif'));
[yfp_621_1,~] = imread(strcat(namepath,'621-1_YFP_20231025.tif'));
[yfp_621_2,~] = imread(strcat(namepath,'621-2_YFP_20231025.tif'));
[yfp_621_3,~] = imread(strcat(namepath,'621-3_YFP_20231025.tif'));

yfp_503 = [mean(mean(yfp_503_1)),mean(mean(yfp_503_2)),mean(mean(yfp_503_3))];
yfp_502 = [mean(mean(yfp_502_1)),mean(mean(yfp_502_2)),mean(mean(yfp_502_3))];
yfp_146 = [mean(mean(yfp_146_1)),mean(mean(yfp_146_2)),mean(mean(yfp_146_3))];
yfp_476 = [mean(mean(yfp_476_1)),mean(mean(yfp_476_2)),mean(mean(yfp_476_3))];
yfp_678 = [mean(mean(yfp_678_1)),mean(mean(yfp_678_2)),mean(mean(yfp_678_3))];
yfp_628 = [mean(mean(yfp_628_1)),mean(mean(yfp_628_2)),mean(mean(yfp_628_3))];
yfp_677 = [mean(mean(yfp_677_1)),mean(mean(yfp_677_2)),mean(mean(yfp_677_3))];
yfp_621 = [mean(mean(yfp_621_1)),mean(mean(yfp_621_2)),mean(mean(yfp_621_3))];

% [qsold_mean,qsold_std]= ev_ratio(yfp_146,yfp_476);
% [qsnew_mean,qsnew_std]= ev_ratio(yfp_677,yfp_621);

%% plot suppfig 1a
clc
fluo_series = [mean(yfp_503), mean(yfp_502) ;mean(yfp_678), mean(yfp_628); mean(yfp_146), mean(yfp_476);...
    mean(yfp_677), mean(yfp_621)]; 
fluo_error = [std(yfp_503), std(yfp_502);std(yfp_678),std(yfp_628); std(yfp_146), std(yfp_476);std(yfp_677), std(yfp_621)]; 


figure
b = bar(fluo_series, 'grouped');
hold on
for k = 1:numel(b)                                                     
    xtips = b(k).XEndPoints;
    ytips = b(k).YEndPoints;
    errorbar(xtips,ytips,fluo_error(:,k), 'k','linestyle','none','LineWidth',2,'CapSize',12)
end
hold off
ylabel('YFP (a.u.)')
XTickLabel={'IS− LD−' ; 'IS− LD+';'IS+ LD−';'IS+ LD+'};
set(gca, 'XTickLabel', XTickLabel);
legend('+ Positive feedback','− Positive feedback')


%% suppfig 1b
%read liquid result
num = xlsread([crt_path  '/experiment data/source data file.xlsx'],'supp fig1b');
data_idx = ~isnan(num(:, 1));
data = num(data_idx,[2:4,7:9]);

conc = [100,30,10,3,1,0.3,0.1,0.01]; % erythromycin
figure
hold on
l2 = shadedErrorBar(conc,transpose(data(1:8,4:6)),{@mean,@std},'lineprops',{'-o','color','#DC582A'}); 
l2 = l2.mainLine;
l3= shadedErrorBar(conc,transpose(data(1:8,1:3)),{@mean,@std},'lineprops',{'-o','color','#ADD8E6'}); 
l3 = l3.mainLine;
l4=shadedErrorBar(conc,transpose(data(9:16,4:6)),{@mean,@std},'lineprops',{'-o','color','#FFC000'}); 
l4 = l4.mainLine;
l5=shadedErrorBar(conc,transpose(data(9:16,1:3)),{@mean,@std},'lineprops',{'-o','color','#0072BD'}); 
l5 = l5.mainLine;

set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
ylabel('YFP(a.u.)')
xlabel('erythromycin conc (ug/ml)')
xticklabels({'0', '10^{-1}', '10^0','10^1', '10^2'})
leg1=legend([l2 l3 l4 l5],'IS− PF−','IS+ PF−','IS− PF+','IS+ PF+','Location','NorthWest');


%% supplementary fig 4b
t = [3.42,3.75,4.05,4.43,4.83,5.3,5.73,6.16,6.55,7.03,8.66,9.4]; % for pmfw476
ptime = [1:12];

allcent = 1;
namepath = [crt_path '/experiment data/changing ery/']
name_annex = '2-';
name_annex2 = '_1-12.mat';
load([namepath name_annex num2str(1) name_annex2])
plate_1_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(2) name_annex2])
plate_2_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_3_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_3_mc = fluo_ave(mc_data_ad,cent_data,allcent);

name_annex = '4-';
load([namepath name_annex num2str(1) name_annex2])
plate_21_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(2) name_annex2])
plate_22_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_23_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

name_annex = '6-';
load([namepath name_annex num2str(1) name_annex2])
plate_31_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(2) name_annex2])
plate_32_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_33_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

m1 = timecourse_distance(plate_1_yfp(:,ptime),0);
m2 = timecourse_distance(plate_2_yfp(:,ptime),0);
m3 = timecourse_distance(plate_3_yfp(:,ptime),0);
m21 = timecourse_distance(plate_21_yfp(:,ptime),0);
m22 = timecourse_distance(plate_22_yfp(:,ptime),0);
m23 = timecourse_distance(plate_23_yfp(:,ptime),0);
m31 = timecourse_distance(plate_31_yfp(:,ptime),0);
m32 = timecourse_distance(plate_32_yfp(:,ptime),0);
m33 = timecourse_distance(plate_33_yfp(:,ptime),0);

xx1 = linspace(3,10,100);
yy1 = 1*xx1.^(1);
yy2 =3/sqrt(3)*xx1.^(1/2);

ptime = [3:12];
figure
box on
%ah = axes;
hold on
l2 = shadedErrorBar(t(3:12),[m1(ptime)*0.01468,m2(ptime)*0.01468,m3(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#bae4b3'}); 
l2 = l2.mainLine;
l3 = shadedErrorBar(t(ptime),[m21(ptime)*0.01468,m22(ptime)*0.01468,m23(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#74c476'}); 
l3 = l3.mainLine;
l4 = shadedErrorBar(t(ptime),[m31(ptime)*0.01468,m32(ptime)*0.01468,m33(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#238b45'}); 
l4 = l4.mainLine;
l5 = plot(xx1,yy1,'k:');
l6 = plot(xx1,yy2,':','color','#808080');
hold off

xlim([3 10]);
ylim([2 20]);
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
yticks([2 4 6 8 10]);
yticklabels({'2', '4', '6','8', '10'})
xlabel('Time [hr]')
ylabel('Wavefront-HWHM [mm]')
leg1=legend([l2 l3 l4], 'ery=2ug','ery=4ug','ery=6ug','Location','NorthWest');
set(leg1,'FontSize',20);
ah1=axes('position',get(gca,'position'),'visible','off');
leg2=legend(ah1,[l5 l6],'\propto t^1','\propto t^{1/2}','Location','SouthEast');set(leg2,'FontSize',20);


%% Supplementary figure 2a growth dynamics for IS-PF- 
% read growth dynamics data
[num, txt,raw] = xlsread([crt_path '/experiment data/source data file.xlsx'],'fig 4 a-b');
data_idx = ~isnan(num(:, 1));
data = num(data_idx,[1,2,5,6,9,10]);

circuit = data(52:78,:);
%data(1:24,:); % IS+PF+
% data(25:51,:); %IS-PF+
% data(52:78,:); %IS-PF-
% data(79:105,:);%IS+PF+

circuit_od2_time = unique(circuit(1:27,1)) ;
circuit_od2_res = reshape(circuit(1:27,2),[3,9]);
circuit_od6_time = unique(circuit(1:27,3)) ;
circuit_od6_res = reshape(circuit(1:27,4),[3,9]);
circuit_od18_time = unique(circuit(1:27,5)) ;
circuit_od18_res = reshape(circuit(1:27,6),[3,9]);

fun2 = @(par,t)(par(2)+par(1)-log(exp(par(2))+(exp(par(1))-exp(par(2)))*exp(-par(3)*(t+log((1+exp(-4*(t-par(4))))/(1+exp(4*par(4))))/4))));

% for od=0.2
par0 = [log(60000),log(30),1,1];
par_citcuit2_f2 = lsqcurvefit(fun2,par0,circuit(1:27,1),log(circuit(1:27,2)));
times = linspace(circuit(1,1),circuit(27,1));
figure
plot(circuit_od2_time,log(mean(circuit_od2_res)),'ko',times,fun2(par_citcuit2_f2,times),'b-')
%set(gca, 'YScale', 'log')
legend('Data','Fitted exponential','location','southeast')

% for od =0.6
par0 = [log(120000),log(100),1,1];
par_citcuit6_f2 = lsqcurvefit(fun2,par0,circuit(1:27,3),log(circuit(1:27,4)));
times = linspace(circuit(1,3),circuit(27,3));
plot(circuit_od6_time,log(mean(circuit_od6_res)),'ko',times,fun2(par_citcuit6_f2,times),'b-')
%set(gca, 'YScale', 'log')
legend('Data','Fitted exponential','location','southeast')

% for od=1.8
par0 = [log(120000),log(500),1,1];
par_citcuit18_f2 = lsqcurvefit(fun2,par0,circuit(1:27,5),log(circuit(1:27,6)));
times = linspace(circuit(1,5),circuit(27,5));
plot(circuit_od18_time,log(mean(circuit_od18_res)),'ko',times,fun2(par_citcuit18_f2,times),'b-')
%set(gca, 'YScale', 'log')
legend('Data','Fitted exponential','location','southeast')

times = linspace(circuit(1,1),6.5);
figure
hold on
box on
plot(circuit(:,1),circuit(:,2),'o','color','#bae4b3')
plot(circuit(:,3),circuit(:,4),'o','color','#74c476')
plot(circuit(:,5),circuit(:,6),'o','color','#238b45')
plot(times,exp(fun2(par_citcuit2_f2,times)),'-','color','#bae4b3')
plot(times,exp(fun2(par_citcuit6_f2,times)),'-','color','#74c476')
plot(times,exp(fun2(par_citcuit18_f2,times)),'-','color','#238b45')
legend('OD=0.2','OD=0.6','OD=1.8','Location','Northwest')
xlabel('Time [hr]')
ylabel('Cell density [CFU]')
xlim([0 8])
ylim([0 80000])
title('IS- PF-')
%set(gca, 'YScale', 'log')

items = {'mu';'division time (min)';'y0 (log)';'ymax (log)';'lambda';'ratio (1/rho)'};
OD2 = [par_citcuit2_f2(3);log(2)/par_citcuit2_f2(3)*60;par_citcuit2_f2(2);par_citcuit2_f2(1);...
    par_citcuit2_f2(4);exp(par_citcuit2_f2(1))/exp(par_citcuit2_f2(2))];
OD6 = [par_citcuit6_f2(3);log(2)/par_citcuit6_f2(3)*60;par_citcuit6_f2(2);par_citcuit6_f2(1);...
    par_citcuit6_f2(4);exp(par_citcuit6_f2(1))/exp(par_citcuit6_f2(2))];
OD18 = [par_citcuit18_f2(3);log(2)/par_citcuit18_f2(3)*60;par_citcuit18_f2(2);par_citcuit18_f2(1);...
    par_citcuit18_f2(4);exp(par_citcuit18_f2(1))/exp(par_citcuit18_f2(2))];

T = table(OD2,OD6,OD18,'RowNames',items)

%% supplementary figure 2b growth dynamics IS-PF+
% read growth dynamics data
[num, txt,raw] = xlsread([crt_path  '/experiment data/source data file.xlsx'],'fig 4 a-b');
data_idx = ~isnan(num(:, 1));
data = num(data_idx,[1,2,5,6,9,10]);

circuit = data(25:51,:); %IS-PF+
%data(1:24,:); % IS+PF+
% data(25:51,:); %IS-PF+
% data(52:78,:); %IS-PF-
% data(79:105,:);%IS+PF-

circuit_od2_time = unique(circuit(1:27,1)) ;
circuit_od2_res = reshape(circuit(1:27,2),[3,9]);
circuit_od6_time = unique(circuit(1:27,3)) ;
circuit_od6_res = reshape(circuit(1:27,4),[3,9]);
circuit_od18_time = unique(circuit(1:27,5)) ;
circuit_od18_res = reshape(circuit(1:27,6),[3,9]);


fun2 = @(par,t)(par(2)+par(1)-log(exp(par(2))+(exp(par(1))-exp(par(2)))*exp(-par(3)*(t+log((1+exp(-4*(t-par(4))))/(1+exp(4*par(4))))/4))));

% for od=0.2
par0 = [log(60000),log(30),1,1];
par_citcuit2_f2 = lsqcurvefit(fun2,par0,circuit(1:27,1),log(circuit(1:27,2)));
times = linspace(circuit(1,1),circuit(27,1));
figure
plot(circuit_od2_time,log(mean(circuit_od2_res)),'ko',times,fun2(par_citcuit2_f2,times),'b-')
%set(gca, 'YScale', 'log')
legend('Data','Fitted exponential','location','southeast')

% for od =0.6
par0 = [log(120000),log(100),1,1];
par_citcuit6_f2 = lsqcurvefit(fun2,par0,circuit(1:27,3),log(circuit(1:27,4)));
times = linspace(circuit(1,3),circuit(27,3));
plot(circuit_od6_time,log(mean(circuit_od6_res)),'ko',times,fun2(par_citcuit6_f2,times),'b-')
%set(gca, 'YScale', 'log')
legend('Data','Fitted exponential','location','southeast')

% for od=1.8
par0 = [log(120000),log(500),1,1];
par_citcuit18_f2 = lsqcurvefit(fun2,par0,circuit(1:27,5),log(circuit(1:27,6)));
times = linspace(circuit(1,5),circuit(27,5));
plot(circuit_od18_time,log(mean(circuit_od18_res)),'ko',times,fun2(par_citcuit18_f2,times),'b-')
%set(gca, 'YScale', 'log')
legend('Data','Fitted exponential','location','southeast')

times = linspace(circuit(1,1),6.5);
figure
hold on
box on
plot(circuit(:,1),circuit(:,2),'o','color','#bae4b3')
plot(circuit(:,3),circuit(:,4),'o','color','#74c476')
plot(circuit(:,5),circuit(:,6),'o','color','#238b45')
plot(times,exp(fun2(par_citcuit2_f2,times)),'-','color','#bae4b3')
plot(times,exp(fun2(par_citcuit6_f2,times)),'-','color','#74c476')
plot(times,exp(fun2(par_citcuit18_f2,times)),'-','color','#238b45')
legend('OD=0.2','OD=0.6','OD=1.8','Location','Northwest')
xlabel('Time [hr]')
ylabel('Cell density [CFU]')
xlim([0 8])
ylim([0 80000])
title('IS- PF+')
%set(gca, 'YScale', 'log')

count_od2= exp(fun2(par_citcuit2_f2,t));
count_od6= exp(fun2(par_citcuit6_f2,t));
count_od18= exp(fun2(par_citcuit18_f2,t));


items = {'mu';'division time (min)';'y0 (log)';'ymax (log)';'lambda';'ratio (1/rho)'};
OD2 = [par_citcuit2_f2(3);log(2)/par_citcuit2_f2(3)*60;par_citcuit2_f2(2);par_citcuit2_f2(1);...
    par_citcuit2_f2(4);exp(par_citcuit2_f2(1))/exp(par_citcuit2_f2(2))];
OD6 = [par_citcuit6_f2(3);log(2)/par_citcuit6_f2(3)*60;par_citcuit6_f2(2);par_citcuit6_f2(1);...
    par_citcuit6_f2(4);exp(par_citcuit6_f2(1))/exp(par_citcuit6_f2(2))];
OD18 = [par_citcuit18_f2(3);log(2)/par_citcuit18_f2(3)*60;par_citcuit18_f2(2);par_citcuit18_f2(1);...
    par_citcuit18_f2(4);exp(par_citcuit18_f2(1))/exp(par_citcuit18_f2(2))];

% T = table(OD2,OD6,OD18,'RowNames',items)
% writetable(T,'fitted twgrowth_new.xlsx','WriteRowNames',true) 

%% supplementary figure 2d growth dynamics for IS+PF+
% read growth dynamics data
[num, txt,raw] = xlsread([crt_path  '/experiment data/source data file.xlsx'],'fig 4 a-b');
data_idx = ~isnan(num(:, 1));
data = num(data_idx,[1,2,5,6,9,10]);

circuit = data(1:24,:); % IS+PF+
circuit_od2_time = unique(circuit(1:24,1)) ;
circuit_od2_res = reshape(circuit(1:24,2),[3,8]);
circuit_od6_time = unique(circuit(1:21,3)) ;
circuit_od6_res = reshape(circuit(1:21,4),[3,7]);
circuit_od18_time = unique(circuit(1:18,5)) ;
circuit_od18_res = reshape(circuit(1:18,6),[3,6]);


fun2 = @(par,t)(par(2)+par(1)-log(exp(par(2))+(exp(par(1))-exp(par(2)))*exp(-par(3)*(t+log((1+exp(-4*(t-par(4))))/(1+exp(4*par(4))))/4))));

% for od=0.2
par0 = [log(60000),log(30),1,1];
par_citcuit2_f2 = lsqcurvefit(fun2,par0,circuit(1:24,1),log(circuit(1:24,2)));
times = linspace(circuit(1,1),circuit(24,1));
figure
plot(circuit_od2_time,log(mean(circuit_od2_res)),'ko',times,fun2(par_citcuit2_f2,times),'b-')
%set(gca, 'YScale', 'log')
legend('Data','Fitted exponential','location','southeast')

% for od =0.6
par0 = [log(120000),log(100),1,1];
par_citcuit6_f2 = lsqcurvefit(fun2,par0,circuit(1:21,3),log(circuit(1:21,4)));
times = linspace(circuit(1,3),circuit(21,3));
plot(circuit_od6_time,log(mean(circuit_od6_res)),'ko',times,fun2(par_citcuit6_f2,times),'b-')
%set(gca, 'YScale', 'log')
legend('Data','Fitted exponential','location','southeast')

% for od=1.8
par0 = [log(120000),log(500),1,1];
par_citcuit18_f2 = lsqcurvefit(fun2,par0,circuit(1:18,5),log(circuit(1:18,6)));
times = linspace(circuit(1,5),circuit(18,5));
plot(circuit_od18_time,log(mean(circuit_od18_res)),'ko',times,fun2(par_citcuit18_f2,times),'b-')
%set(gca, 'YScale', 'log')
legend('Data','Fitted exponential','location','southeast')

times = linspace(circuit(1,1),6.5);
figure
hold on
box on
plot(circuit(:,1),circuit(:,2),'o','color','#bae4b3')
plot(circuit(:,3),circuit(:,4),'o','color','#74c476')
plot(circuit(:,5),circuit(:,6),'o','color','#238b45')
plot(times,exp(fun2(par_citcuit2_f2,times)),'-','color','#bae4b3')
plot(times,exp(fun2(par_citcuit6_f2,times)),'-','color','#74c476')
plot(times,exp(fun2(par_citcuit18_f2,times)),'-','color','#238b45')
legend('OD=0.2','OD=0.6','OD=1.8','Location','Northwest')
xlabel('Time [hr]')
ylabel('Cell density [CFU]')
xlim([0 8])
ylim([0 80000])
title('IS+ PF+')
%set(gca, 'YScale', 'log')

items = {'mu';'division time (min)';'y0 (log)';'ymax (log)';'lambda';'ratio (1/rho)'};
OD2 = [par_citcuit2_f2(3);log(2)/par_citcuit2_f2(3)*60;par_citcuit2_f2(2);par_citcuit2_f2(1);...
    par_citcuit2_f2(4);exp(par_citcuit2_f2(1))/exp(par_citcuit2_f2(2))];
OD6 = [par_citcuit6_f2(3);log(2)/par_citcuit6_f2(3)*60;par_citcuit6_f2(2);par_citcuit6_f2(1);...
    par_citcuit6_f2(4);exp(par_citcuit6_f2(1))/exp(par_citcuit6_f2(2))];
OD18 = [par_citcuit18_f2(3);log(2)/par_citcuit18_f2(3)*60;par_citcuit18_f2(2);par_citcuit18_f2(1);...
    par_citcuit18_f2(4);exp(par_citcuit18_f2(1))/exp(par_citcuit18_f2(2))];

T = table(OD2,OD6,OD18,'RowNames',items)

%% supplementary figure 5 for IS-PF+ circuits

t = [3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10];

ptime = [1:15];
namepath=namepath3;%namepath3 for IS-PF+
allcent = 1;
name_annex = '2-';
name_annex2 = '_1-15.mat';
name_annex3 = '0_';
load([namepath name_annex num2str(1) name_annex2])
plate_1_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_1_mc = fluo_ave(mc_data_ad,cent_data,allcent);


load([namepath name_annex num2str(2) name_annex2])
plate_2_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_2_mc = fluo_ave(mc_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_3_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_3_mc = fluo_ave(mc_data_ad,cent_data,allcent);

name_annex = '6-';
load([namepath name_annex num2str(1) name_annex2])
plate_21_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_21_mc = fluo_ave(mc_data_ad,cent_data,allcent);

load([namepath name_annex num2str(2) name_annex2])
plate_22_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_22_mc = fluo_ave(mc_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_23_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_23_mc = fluo_ave(mc_data_ad,cent_data,allcent);

name_annex = '18-';
load([namepath name_annex num2str(1) name_annex2])
plate_31_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_31_mc = fluo_ave(mc_data_ad,cent_data,allcent);

load([namepath name_annex num2str(2) name_annex2])
plate_32_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_32_mc = fluo_ave(mc_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_33_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_33_mc = fluo_ave(mc_data_ad,cent_data,allcent);
plate_33_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);


% calculate distance with half maximum
m1 = timecourse_distance(plate_1_yfp(:,ptime),0);
m2 = timecourse_distance(plate_2_yfp(:,ptime),0);
m3 = timecourse_distance(plate_3_yfp(:,ptime),0);
m21 = timecourse_distance(plate_21_yfp(:,ptime),0);
m22 = timecourse_distance(plate_22_yfp(:,ptime),0);
m23 = timecourse_distance(plate_23_yfp(:,ptime),0);
m31 = timecourse_distance(plate_31_yfp(:,ptime),0);
m32 = timecourse_distance(plate_32_yfp(:,ptime),0);
m33 = timecourse_distance(plate_33_yfp(:,ptime),0);

% reference line
xx1 = linspace(3,10,100);
yy1 = 3/3*xx1.^(1);
yy2 =3/sqrt(3)*xx1.^(1/2);

figure
box on
hold on
l2 = shadedErrorBar(t(ptime),[m1( ptime)*0.01468,m2(ptime)*0.01468,m3(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#bae4b3'}); 
l2 = l2.mainLine;
l3 = shadedErrorBar(t(ptime),[m21(ptime)*0.01468,m22(ptime)*0.01468,m23(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#74c476'}); 
l3 = l3.mainLine;
l4 = shadedErrorBar(t(ptime),[m31(ptime)*0.01468,m32(ptime)*0.01468,m33(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#238b45'}); 
l4 = l4.mainLine;
l5 = plot(xx1,yy1,'k:');
l6 = plot(xx1,yy2,':','color','#808080');
hold off
title('IS-PF+')

xlim([3 10]);
ylim([2 20]);
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
yticks([2 4 6 8 10]);
yticklabels({'2', '4', '6','8', '10'})
xlabel('Time [hr]')
ylabel('Wavefront-HWHM [mm]')
leg1=legend([l2 l3 l4], 'OD=0.2','OD=0.6','OD=1.8','Location','NorthWest');
set(leg1,'FontSize',20);
ah1=axes('position',get(gca,'position'),'visible','off');
leg2=legend(ah1,[l5 l6],'\propto t^1','\propto t^{1/2}','Location','SouthEast');set(leg2,'FontSize',20);

%% threshold Supplementary figure 7b
th=0.1;
m1 = timecourse_distance_thd(plate_1_yfp(:,ptime)./count_od2,th,0);
m2 = timecourse_distance_thd(plate_2_yfp(:,ptime)./count_od2,th,0);
m3 = timecourse_distance_thd(plate_3_yfp(:,ptime)./count_od2,th,0);
m21 = timecourse_distance_thd(plate_21_yfp(:,ptime)./count_od6,th,0);
m22 = timecourse_distance_thd(plate_22_yfp(:,ptime)./count_od6,th,0);
m23 = timecourse_distance_thd(plate_23_yfp(:,ptime)./count_od6,th,0);
m31 = timecourse_distance_thd(plate_31_yfp(:,ptime)./count_od18,th,0);
m32 = timecourse_distance_thd(plate_32_yfp(:,ptime)./count_od18,th,0);
m33 = timecourse_distance_thd(plate_33_yfp(:,ptime)./count_od18,th,0);

xx1 = linspace(3,10,100);
yy1 = 3/3*xx1.^(1);
yy2 =3/sqrt(3)*xx1.^(1/2);

ptime = [2:15];
figure
box on
hold on
l2 = shadedErrorBar(t(ptime),[m1(ptime)*0.01468,m2(ptime)*0.01468,m3(ptime)*0.01468]',{@mean,@std},'lineprops',{'--o','color','#bae4b3'}); 
l2 = l2.mainLine;
l3 = shadedErrorBar(t(ptime),[m21(ptime)*0.01468,m22(ptime)*0.01468,m23(ptime)*0.01468]',{@mean,@std},'lineprops',{'--o','color','#74c476'}); 
l3 = l3.mainLine;
l4 = shadedErrorBar(t(ptime),[m31(ptime)*0.01468,m32(ptime)*0.01468,m33(ptime)*0.01468]',{@mean,@std},'lineprops',{'--o','color','#238b45'}); 
l4 = l4.mainLine;
l5 = plot(xx1,yy1,'k:');
l6 = plot(xx1,yy2,':','color','#808080');
hold off
title('IS- PF+, th=0.1')

xlim([3 10]);
ylim([2 20]);
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
yticks([2 4 6 8 10]);
yticklabels({'2', '4', '6','8', '10'})
xlabel('Time [hr]')
ylabel('Threshold [mm]')
leg1=legend([l2 l3 l4], 'OD=0.2','OD=0.6','OD=1.8','Location','NorthWest');
set(leg1,'FontSize',20);
ah1=axes('position',get(gca,'position'),'visible','off');
leg2=legend(ah1,[l5 l6],'\propto t^1','\propto t^{1/2}','Location','SouthEast');set(leg2,'FontSize',20);

%% %% supplementary figure 5a growth dynamics IS+PF-
% read growth dynamics data
[num, txt,raw] = xlsread([crt_path  '/experiment data/source data file.xlsx'],'fig 4 a-b');
data_idx = ~isnan(num(:, 1));
data = num(data_idx,[1,2,5,6,9,10]);

circuit = data(79:105,:); %IS+PF-

circuit_od2_time = unique(circuit(1:27,1)) ;
circuit_od2_res = reshape(circuit(1:27,2),[3,9]);
circuit_od6_time = unique(circuit(1:27,3)) ;
circuit_od6_res = reshape(circuit(1:27,4),[3,9]);
circuit_od18_time = unique(circuit(1:27,5)) ;
circuit_od18_res = reshape(circuit(1:27,6),[3,9]);


fun2 = @(par,t)(par(2)+par(1)-log(exp(par(2))+(exp(par(1))-exp(par(2)))*exp(-par(3)*(t+log((1+exp(-4*(t-par(4))))/(1+exp(4*par(4))))/4))));

% for od=0.2
par0 = [log(60000),log(30),1,1];
par_citcuit2_f2 = lsqcurvefit(fun2,par0,circuit(1:27,1),log(circuit(1:27,2)));
times = linspace(circuit(1,1),circuit(27,1));
figure
plot(circuit_od2_time,log(mean(circuit_od2_res)),'ko',times,fun2(par_citcuit2_f2,times),'b-')
%set(gca, 'YScale', 'log')
legend('Data','Fitted exponential','location','southeast')

% for od =0.6
par0 = [log(120000),log(100),1,1];
par_citcuit6_f2 = lsqcurvefit(fun2,par0,circuit(1:27,3),log(circuit(1:27,4)));
times = linspace(circuit(1,3),circuit(27,3));
plot(circuit_od6_time,log(mean(circuit_od6_res)),'ko',times,fun2(par_citcuit6_f2,times),'b-')
%set(gca, 'YScale', 'log')
legend('Data','Fitted exponential','location','southeast')

% for od=1.8
par0 = [log(120000),log(500),1,1];
par_citcuit18_f2 = lsqcurvefit(fun2,par0,circuit(1:27,5),log(circuit(1:27,6)));
times = linspace(circuit(1,5),circuit(27,5));
plot(circuit_od18_time,log(mean(circuit_od18_res)),'ko',times,fun2(par_citcuit18_f2,times),'b-')
%set(gca, 'YScale', 'log')
legend('Data','Fitted exponential','location','southeast')

times = linspace(circuit(1,1),6.5);
figure
hold on
box on
plot(circuit(:,1),circuit(:,2),'o','color','#bae4b3')
plot(circuit(:,3),circuit(:,4),'o','color','#74c476')
plot(circuit(:,5),circuit(:,6),'o','color','#238b45')
plot(times,exp(fun2(par_citcuit2_f2,times)),'-','color','#bae4b3')
plot(times,exp(fun2(par_citcuit6_f2,times)),'-','color','#74c476')
plot(times,exp(fun2(par_citcuit18_f2,times)),'-','color','#238b45')
legend('OD=0.2','OD=0.6','OD=1.8','Location','Northwest')
xlabel('Time [hr]')
ylabel('Cell density [CFU]')
xlim([0 8])
ylim([0 80000])
title('IS+ PF-')
%set(gca, 'YScale', 'log')

count_od2= exp(fun2(par_citcuit2_f2,t));
count_od6= exp(fun2(par_citcuit6_f2,t));
count_od18= exp(fun2(par_citcuit18_f2,t));


items = {'mu';'division time (min)';'y0 (log)';'ymax (log)';'lambda';'ratio (1/rho)'};
OD2 = [par_citcuit2_f2(3);log(2)/par_citcuit2_f2(3)*60;par_citcuit2_f2(2);par_citcuit2_f2(1);...
    par_citcuit2_f2(4);exp(par_citcuit2_f2(1))/exp(par_citcuit2_f2(2))];
OD6 = [par_citcuit6_f2(3);log(2)/par_citcuit6_f2(3)*60;par_citcuit6_f2(2);par_citcuit6_f2(1);...
    par_citcuit6_f2(4);exp(par_citcuit6_f2(1))/exp(par_citcuit6_f2(2))];
OD18 = [par_citcuit18_f2(3);log(2)/par_citcuit18_f2(3)*60;par_citcuit18_f2(2);par_citcuit18_f2(1);...
    par_citcuit18_f2(4);exp(par_citcuit18_f2(1))/exp(par_citcuit18_f2(2))];

% T = table(OD2,OD6,OD18,'RowNames',items)
% writetable(T,'fitted twgrowth_new.xlsx','WriteRowNames',true) 

t = [3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10];

ptime = [1:15];

allcent = 1;
namepath = namepath2;% namethpath2 for IS+PF-
name_annex = '621-2-';
name_annex2 = '_1-15.mat';
name_annex3 = '0_';
load([namepath name_annex num2str(1) name_annex2])
plate_1_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_1_mc = fluo_ave(mc_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(1) name_annex2])
plate_1_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(2) name_annex2])
plate_2_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_2_mc = fluo_ave(mc_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(2) name_annex2])
plate_2_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_3_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_3_mc = fluo_ave(mc_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(3) name_annex2])
plate_3_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

name_annex = '621-6-';
load([namepath name_annex num2str(1) name_annex2])
plate_21_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_21_mc = fluo_ave(mc_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(1) name_annex2])
plate_21_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(2) name_annex2])

plate_22_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_22_mc = fluo_ave(mc_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(2) name_annex2])
plate_22_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_23_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_23_mc = fluo_ave(mc_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(3) name_annex2])
plate_23_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

name_annex = '621-18-';
load([namepath name_annex num2str(1) name_annex2])
plate_31_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_31_mc = fluo_ave(mc_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(1) name_annex2])
plate_31_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(2) name_annex2])
plate_32_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_32_mc = fluo_ave(mc_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(2) name_annex2])
plate_32_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_33_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_33_mc = fluo_ave(mc_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(3) name_annex2])
plate_33_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

%% calculate distance with half maximum 
m1 = timecourse_distance(plate_1_yfp(:,ptime)-plate_1_yfp_bg(:,ptime),1);
m2 = timecourse_distance(plate_2_yfp(:,ptime)-plate_2_yfp_bg(:,ptime),1);
m3 = timecourse_distance(plate_3_yfp(:,ptime)-plate_3_yfp_bg(:,ptime),1);
m21 = timecourse_distance(plate_21_yfp(:,ptime)-plate_21_yfp_bg(:,ptime),1);
m22 = timecourse_distance(plate_22_yfp(:,ptime)-plate_22_yfp_bg(:,ptime),1);
m23 = timecourse_distance(plate_23_yfp(:,ptime)-plate_23_yfp_bg(:,ptime),1);
m31 = timecourse_distance(plate_31_yfp(:,ptime)-plate_31_yfp_bg(:,ptime),1);
m32 = timecourse_distance(plate_32_yfp(:,ptime)-plate_32_yfp_bg(:,ptime),1);
m33 = timecourse_distance(plate_33_yfp(:,ptime)-plate_33_yfp_bg(:,ptime),1);

% reference line
xx1 = linspace(3,10,100);
yy1 = 3/3*xx1.^(1);
yy2 =3/sqrt(3)*xx1.^(1/2);

figure
box on
hold on
l2 = shadedErrorBar(t(ptime),[m1( ptime)*0.01468,m2(ptime)*0.01468,m3(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#bae4b3'}); 
l2 = l2.mainLine;
l3 = shadedErrorBar(t(ptime),[m21(ptime)*0.01468,m22(ptime)*0.01468,m23(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#74c476'}); 
l3 = l3.mainLine;
l4 = shadedErrorBar(t(ptime),[m31(ptime)*0.01468,m32(ptime)*0.01468,m33(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#238b45'}); 
l4 = l4.mainLine;
l5 = plot(xx1,yy1,'k:');
l6 = plot(xx1,yy2,':','color','#808080');
hold off

xlim([3 10]);
ylim([2 20]);
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
yticks([2 4 6 8 10]);
yticklabels({'2', '4', '6','8', '10'})
xlabel('Time [hr]')
ylabel('Wavefront-HWHM [mm]')
leg1=legend([l2 l3 l4], 'OD=0.2','OD=0.6','OD=1.8','Location','NorthWest');
set(leg1,'FontSize',20);
ah1=axes('position',get(gca,'position'),'visible','off');
leg2=legend(ah1,[l5 l6],'\propto t^1','\propto t^{1/2}','Location','SouthEast');set(leg2,'FontSize',20);

%% Supplementary figure 7b
th=0.1;
ptime = [1:15];

[m1,thd1] = timecourse_distance_thd((plate_1_yfp(:,ptime)-plate_1_yfp_bg(:,ptime))./count_od2,th,1);
[m2,thd2] = timecourse_distance_thd((plate_2_yfp(:,ptime)-plate_2_yfp_bg(:,ptime))./count_od2,th,1);
[m3,thd3] = timecourse_distance_thd((plate_3_yfp(:,ptime)-plate_3_yfp_bg(:,ptime))./count_od2,th,1);
[m21,thd21] = timecourse_distance_thd((plate_21_yfp(:,ptime)-plate_21_yfp_bg(:,ptime))./count_od6,th,1);
[m22,thd22] = timecourse_distance_thd((plate_22_yfp(:,ptime)-plate_22_yfp_bg(:,ptime))./count_od6,th,1);
[m23,thd23] = timecourse_distance_thd((plate_23_yfp(:,ptime)-plate_23_yfp_bg(:,ptime))./count_od6,th,1);
[m31,thd31] = timecourse_distance_thd((plate_31_yfp(:,ptime)-plate_31_yfp_bg(:,ptime))./count_od18,th,1);
[m32,thd32] = timecourse_distance_thd((plate_32_yfp(:,ptime)-plate_32_yfp_bg(:,ptime))./count_od18,th,1);
[m33,thd33] = timecourse_distance_thd((plate_33_yfp(:,ptime)-plate_33_yfp_bg(:,ptime))./count_od18,th,1);


xx1 = linspace(3,10,100);
yy1 = 3/3*xx1.^(1);
yy2 =3/sqrt(3)*xx1.^(1/2);

ptime = [2:15];
figure
box on
hold on
l2 = shadedErrorBar(t(ptime),[m1(ptime)*0.01468,m2(ptime)*0.01468,m3(ptime)*0.01468]',{@mean,@std},'lineprops',{'--o','color','#bae4b3'}); 
l2 = l2.mainLine;
l3 = shadedErrorBar(t(ptime),[m21(ptime)*0.01468,m22(ptime)*0.01468,m23(ptime)*0.01468]',{@mean,@std},'lineprops',{'--o','color','#74c476'}); 
l3 = l3.mainLine;
l4 = shadedErrorBar(t(ptime),[m31(ptime)*0.01468,m32(ptime)*0.01468,m33(ptime)*0.01468]',{@mean,@std},'lineprops',{'--o','color','#238b45'}); 
l4 = l4.mainLine;
l5 = plot(xx1,yy1,'k:');
l6 = plot(xx1,yy2,':','color','#808080');
hold off
title('IS+ PF-, th=0.1')

xlim([3 10]);
ylim([2 20]);
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
yticks([2 4 6 8 10]);
yticklabels({'2', '4', '6','8', '10'})
xlabel('Time [hr]')
ylabel('Threshold [mm]')
leg1=legend([l2 l3 l4], 'OD=0.2','OD=0.6','OD=1.8','Location','NorthWest');
set(leg1,'FontSize',20);
ah1=axes('position',get(gca,'position'),'visible','off');
leg2=legend(ah1,[l5 l6],'\propto t^1','\propto t^{1/2}','Location','SouthEast');set(leg2,'FontSize',20);

%% supplementary fig6 

t = [3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10];

ptime = [1:15];

allcent = 1;
name_annex = '6-';
name_annex2 = '_1-15.mat';
name_annex3 = '0_';
load([namepath1 name_annex num2str(1) name_annex2])
plate_1_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath1 name_annex num2str(2) name_annex2])
plate_2_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath1 name_annex num2str(3) name_annex2])
plate_3_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);


name_annex = '6-';
load([namepath3 name_annex num2str(1) name_annex2])
plate_31_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath3 name_annex num2str(2) name_annex2])
plate_32_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath3 name_annex num2str(3) name_annex2])
plate_33_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

name_annex = '621-6-';
load([namepath2 name_annex num2str(1) name_annex2])
plate_21_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath2 name_annex name_annex3 num2str(1) name_annex2])
plate_21_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath2 name_annex num2str(2) name_annex2])
plate_22_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath2 name_annex name_annex3 num2str(2) name_annex2])
plate_22_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath2 name_annex num2str(3) name_annex2])
plate_23_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath2 name_annex name_annex3 num2str(3) name_annex2])
plate_23_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

name_annex = '677-6-';
load([namepath4 name_annex num2str(1) name_annex2])
plate_41_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_41_mc = fluo_ave(mc_data_ad,cent_data,allcent);
load([namepath4 name_annex name_annex3 num2str(1) name_annex2])
plate_41_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath4 name_annex num2str(2) name_annex2])
plate_42_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_42_mc = fluo_ave(mc_data_ad,cent_data,allcent);
load([namepath4 name_annex name_annex3 num2str(2) name_annex2])
plate_42_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath4 name_annex num2str(3) name_annex2])
plate_43_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_43_mc = fluo_ave(mc_data_ad,cent_data,allcent);
load([namepath4 name_annex name_annex3 num2str(3) name_annex2])
plate_43_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);


[num, txt,raw] = xlsread([crt_path '/experiment data/source data file.xlsx'],'fig 4 a-b');
data_idx = ~isnan(num(:, 1));
data = num(data_idx,[1,2,5,6,9,10]);

tw = data(1:24,:);
pf = data(25:51,:);
neg = data(52:78,:);
qs = data(79:105,:);

tw_time = unique(tw(1:21,3)) ;
tw_od6_res = reshape(tw(1:21,4),[3,7]);
pf_time = unique(pf(1:27,3)) ;
pf_od6_res = reshape(pf(1:27,4),[3,9]);
neg_time = unique(neg(1:27,3)) ;
neg_od6_res = reshape(neg(1:27,4),[3,9]);
qs_time = unique(qs(1:27,3)) ;
qs_od6_res = reshape(qs(1:27,4),[3,9]);

fun2 = @(par,t)(par(2)+par(1)-log(exp(par(2))+(exp(par(1))-exp(par(2)))*exp(-par(3)*(t+log((1+exp(-4*(t-par(4))))/(1+exp(4*par(4))))/4))));
par0 = [log(100000),log(100),1,1];

par_tw6_new = lsqcurvefit(fun2,par0,tw(1:21,3),log(tw(1:21,4)));
times = linspace(tw(1,3),tw(21,3));
figure
hold on
plot(tw_time,log(mean(tw_od6_res)),'ko',times,fun2(par_tw6_new,times),'b-')
legend('Data','model-lag','model-no lag','location','southeast')
xlabel('time (hour)')
ylabel('log(cell density)')
title('IS+ PF+')

par_pf6_new = lsqcurvefit(fun2,par0,pf(1:27,3),log(pf(1:27,4)));
times = linspace(pf(1,3),pf(27,3));
figure
hold on
plot(pf_time,log(mean(pf_od6_res)),'ko',times,fun2(par_pf6_new,times),'b-')
legend('Data','model-no lag','model-lag','location','southeast')
xlabel('time (hour)')
ylabel('log(cell density)')
title('IS- PF+')

par_neg6_new = lsqcurvefit(fun2,par0,neg(1:27,3),log(neg(1:27,4)));
times = linspace(neg(1,3),neg(27,3));
figure 
plot(neg_time,log(mean(neg_od6_res)),'ko',times,fun2(par_neg6_new,times),'b-')
legend('Data','model-no lag','model-lag','location','southeast')
xlabel('time (hour)')
ylabel('log(cell density)')
title('IS- PF-')


par_qs6_new = lsqcurvefit(fun2,par0,qs(1:27,3),log(qs(1:27,4)));
times = linspace(qs(1,3),qs(27,3));
figure
plot(qs_time,log(mean(qs_od6_res)),'ko',times,fun2(par_qs6_new,times),'b-')
legend('Data','model-no lag','model-lag','location','southeast')
xlabel('time (hour)')
ylabel('log(cell density)')
title('IS+ PF-')

cell_tw = exp(fun2(par_tw6_new,t));
cell_pf = exp(fun2(par_pf6_new,t));
cell_neg = exp(fun2(par_neg6_new,t));
cell_qs = exp(fun2(par_qs6_new,t));

%% changing threshold 
ptime =[1:15];

th=0.4;
[m1,thd1] = timecourse_distance_thd((plate_1_yfp(:,ptime))./cell_neg,th,0);
[m2,thd2] = timecourse_distance_thd((plate_2_yfp(:,ptime))./cell_neg,th,0);
[m3,thd3] = timecourse_distance_thd((plate_3_yfp(:,ptime))./cell_neg,th,0);
[m21,thd21] = timecourse_distance_thd((plate_21_yfp(:,ptime)-plate_21_yfp_bg(:,ptime))./cell_qs,th,1);
[m22,thd22] = timecourse_distance_thd((plate_22_yfp(:,ptime)-plate_22_yfp_bg(:,ptime))./cell_qs,th,1);
[m23,thd23] = timecourse_distance_thd((plate_23_yfp(:,ptime)-plate_23_yfp_bg(:,ptime))./cell_qs,th,1);
[m31,thd31] = timecourse_distance_thd((plate_31_yfp(:,ptime))./cell_pf,th,0);
[m32,thd32] = timecourse_distance_thd((plate_32_yfp(:,ptime))./cell_pf,th,0);
[m33,thd33] = timecourse_distance_thd((plate_33_yfp(:,ptime))./cell_pf,th,0);
[m41,thd41] = timecourse_distance_thd((plate_41_yfp(:,ptime)-plate_41_yfp_bg(:,ptime))./cell_tw,th,1);
[m42,thd42] = timecourse_distance_thd((plate_42_yfp(:,ptime)-plate_42_yfp_bg(:,ptime))./cell_tw,th,1);
[m43,thd43] = timecourse_distance_thd((plate_43_yfp(:,ptime)-plate_43_yfp_bg(:,ptime))./cell_tw,th,1);

xx1 = linspace(3,10,100);
yy1 = 1*xx1.^(1);
yy2 =3/sqrt(3)*xx1.^(1/2);

figure
box on
ptime = [2:15];
hold on
l2 = shadedErrorBar(t(ptime),[m1(ptime)*0.01468,m2(ptime)*0.01468,m3(ptime)*0.01468]',{@mean,@std},'lineprops',{'--o','color','#DC582A'}); 
l2 = l2.mainLine;
l3= shadedErrorBar(t(ptime),[m21(ptime)*0.01468,m22(ptime)*0.01468,m23(ptime)*0.01468]',{@mean,@std},'lineprops',{'--o','color','#ADD8E6'}); 
l3 = l3.mainLine;
l4=shadedErrorBar(t(ptime),[m31(ptime)*0.01468,m32(ptime)*0.01468,m33(ptime)*0.01468]',{@mean,@std},'lineprops',{'--o','color','#FFC000'}); 
l4 = l4.mainLine;
l5=shadedErrorBar(t(ptime),[m41(ptime)*0.01468,m42(ptime)*0.01468,m43(ptime)*0.01468]',{@mean,@std},'lineprops',{'--o','color','#0072BD'}); 
l5 = l5.mainLine;
%l11 = plot(t(ptime),m_dye(ptime)*0.01468,'r');
l6 = plot(xx1,yy1,'k:');
l7 = plot(xx1,yy2,':','color','#808080');
hold off

xlim([3 10]);
ylim([2 20]);
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
yticks([2 4 6 8 10]);
yticklabels({'2', '4', '6','8', '10'})
xlabel('Time[hr]')
%ylabel('Wavefront-HWHM [mm]')
ylabel('Wavefront-Threshold[mm]')
%title('th=0.5 model-lag')
% leg1=legend([l2 l3 l4 l5],'QS- PF-','QS+ PF-','QS- PF+','Synthetic trigger wave','Location','NorthWest');
leg1=legend([l2 l3 l4 l5],'IS- PF-','IS+ PF-','IS- PF+','IS+ PF+','Location','NorthWest');
set(leg1,'FontSize',20);
ah1=axes('position',get(gca,'position'),'visible','off');
leg2=legend(ah1,[l6 l7],'\propto t^1','\propto t^{1/2}','Location','SouthEast');set(leg2,'FontSize',20);


