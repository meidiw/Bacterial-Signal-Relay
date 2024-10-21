clear all
close all

setFigDef()

%%
crt_path = cd();
namepath1 = [crt_path '/experiment data/IS-PF-/'];
namepath2 = [crt_path '/experiment data/IS+PF-/'];
namepath3 = [crt_path '/experiment data/IS-PF+/'];
namepath4 = [crt_path '/experiment data/IS+PF+/'];
%%
t = [3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10];
ptime = [1:15];

xx1 = linspace(3,10,100);
yy1 = 1*xx1.^(1);
yy2 =3/sqrt(3)*xx1.^(1/2);

t = [3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10];

ptime = [1:15];
allcent = 1;
name_annex = '6-';
name_annex2 = '_1-15.mat';
name_annex3 = '0_';
load([namepath1 name_annex num2str(1) name_annex2])
plate_1_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_1_mc = fluo_ave(mc_data_ad,cent_data,allcent);

load([namepath1 name_annex num2str(2) name_annex2])
plate_2_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_2_mc = fluo_ave(mc_data_ad,cent_data,allcent);

load([namepath1 name_annex num2str(3) name_annex2])
plate_3_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_3_mc = fluo_ave(mc_data_ad,cent_data,allcent);


name_annex = '6-';
load([namepath3 name_annex num2str(1) name_annex2])
plate_31_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_31_mc = fluo_ave(mc_data_ad,cent_data,allcent);

load([namepath3 name_annex num2str(2) name_annex2])
plate_32_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_32_mc = fluo_ave(mc_data_ad,cent_data,allcent);

load([namepath3 name_annex num2str(3) name_annex2])
plate_33_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_33_mc = fluo_ave(mc_data_ad,cent_data,allcent);


name_annex = '621-6-';
load([namepath2 name_annex num2str(1) name_annex2])
plate_21_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_21_mc = fluo_ave(mc_data_ad,cent_data,allcent);
load([namepath2 name_annex name_annex3 num2str(1) name_annex2])
plate_21_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath2 name_annex num2str(2) name_annex2])
plate_22_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_22_mc = fluo_ave(mc_data_ad,cent_data,allcent);
load([namepath2 name_annex name_annex3 num2str(2) name_annex2])
plate_22_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath2 name_annex num2str(3) name_annex2])
plate_23_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
plate_23_mc = fluo_ave(mc_data_ad,cent_data,allcent);
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


%% fig 2c dye and tw 
m1_dye = timecourse_distance(plate_1_mc(:,ptime),0);
m2_dye = timecourse_distance(plate_2_mc(:,ptime),0);
m3_dye = timecourse_distance(plate_3_mc(:,ptime),0);
m21_dye = timecourse_distance(plate_1_mc(:,ptime),0);
m22_dye = timecourse_distance(plate_2_mc(:,ptime),0);
m23_dye = timecourse_distance(plate_3_mc(:,ptime),0);
m31_dye = timecourse_distance(plate_1_mc(:,ptime),0);
m32_dye = timecourse_distance(plate_2_mc(:,ptime),0);
m33_dye = timecourse_distance(plate_3_mc(:,ptime),0);
m41_dye = timecourse_distance(plate_41_mc(:,ptime),0);
m42_dye = timecourse_distance(plate_42_mc(:,ptime),0);
m43_dye = timecourse_distance(plate_43_mc(:,ptime),0);
m41 = timecourse_distance(plate_41_yfp(:,ptime)-plate_41_yfp_bg(:,ptime),1);
m42 = timecourse_distance(plate_42_yfp(:,ptime)-plate_42_yfp_bg(:,ptime),1);
m43 = timecourse_distance(plate_43_yfp(:,ptime)-plate_43_yfp_bg(:,ptime),1);

figure
box on
ptime = [1:15];
hold on
l1 = shadedErrorBar(t(ptime),[m1_dye(ptime)*0.01468,m2_dye(ptime)*0.01468,m3_dye(ptime)*0.01468,...
    m21_dye(ptime)*0.01468,m22_dye(ptime)*0.01468,m23_dye(ptime)*0.01468,...
    m31_dye(ptime)*0.01468,m32_dye(ptime)*0.01468,m33_dye(ptime)*0.01468,...
    m41_dye(ptime)*0.01468,m42_dye(ptime)*0.01468,m43_dye(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#828282'});   
l1 = l1.mainLine;
l2 = shadedErrorBar(t(ptime),[m41(ptime)*0.01468,m42(ptime)*0.01468,m43(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#0072BD'}); 
l2 = l2.mainLine;
l3 = plot(xx1,yy1,'k:');
l4 = plot(xx1,yy2,':','color','#808080');
xlim([3 10]);
ylim([3 20]);
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
yticks([2 4 6 8 10]);
yticklabels({'2', '4', '6','8', '10'})
xlabel('Time [hr]')
ylabel('Wavefront-HWHM [mm]')
leg1=legend([l1 l2 ], 'Dye','IS+ PF+','Location','NorthWest');
set(leg1,'FontSize',20);
ah1=axes('position',get(gca,'position'),'visible','off');
leg2=legend(ah1,[l3 l4],'\propto t^1','\propto t^{1/2}','Location','SouthEast');set(leg2,'FontSize',20);

%% fig 3a 
% read original image to create montage
allcent = 1;
total = [1:15];
ntotal = length(total);
name1 = cell(ntotal,1);
ptime = [1:15];
t = 2000/990.*[50,149,248,347,446,545,644,743,842,941];
frame = [1:10];
frame = flip(frame);

namepath1 = [crt_path '/experiment data/IS-PF-/'];
name_annex = '6-2-t';
name_annex2 = '_1-15.mat';
for i =1:ntotal
    name1{i,1} = [name_annex num2str(total(i)) '_YFP.tif'];
end

load([namepath1 name_annex(1:end-2) name_annex2])
cent_data1 = cent_data;
plate_1_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
m1 = timecourse_distance(plate_1_yfp(:,ptime),0);
[negmin,negmax]=find_minmax(plate_1_yfp);
negmin = min(negmin);
negmax = max(negmax);

namepath2 = [crt_path '/experiment data/IS+PF-/'];
name_annex = '621-6-1-t';
name_annex3 = '0_';
name2 = cell(ntotal,1);
for i =1:ntotal
    name2{i,1} = [name_annex num2str(total(i)) '_YFP.tif'];
end
load([namepath2 name_annex(1:end-2) name_annex2])
cent_data2 = cent_data;
plate_2_yfp = fluo_ave(YFP_data_ad,cent_data2,allcent);
load([namepath2 name_annex(1:end-3) name_annex3 num2str(1) name_annex2])
plate_2_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);
m2 = timecourse_distance(plate_2_yfp(:,ptime)-plate_2_yfp_bg(:,ptime),1);
[qsmin,qsmax] = find_minmax(plate_2_yfp);
qsmin = min(qsmin);
qsmax = max(qsmax);

namepath3 = [crt_path '/experiment data/IS-PF+/'];
name_annex = '6-1-t';
name3 = cell(ntotal,1);
for i =1:ntotal
    name3{i,1} = [name_annex num2str(total(i)) '_YFP.tif'];
end
load([namepath3 name_annex(1:end-2) name_annex2])
cent_data3 = cent_data;
plate_3_yfp = fluo_ave(YFP_data_ad,cent_data3,allcent);
m3 = timecourse_distance(plate_3_yfp(:,ptime),0);
[pfmin,pfmax]=find_minmax(plate_3_yfp);
pfmin = min(pfmin);
pfmax = max(pfmax);

namepath4 = [crt_path '/experiment data/IS+PF+/'];
name_annex = '677-6-1-t';
name4 = cell(ntotal,1);
for i =1:ntotal
    name4{i,1} = [name_annex num2str(total(i)) '_YFP.tif'];
end
load([namepath4 name_annex(1:end-2) name_annex2])
cent_data4 = cent_data;
plate_4_yfp = fluo_ave(YFP_data_ad,cent_data4,allcent);
load([namepath4 name_annex(1:end-3) name_annex3 num2str(1) name_annex2])
plate_4_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);
m4 = timecourse_distance(plate_4_yfp(:,ptime)-plate_4_yfp_bg(:,ptime),1);
[twmin,twmax]= find_minmax(plate_4_yfp);
twmin = min(twmin);
twmax = max(twmax);

allmin = 500; %overall generization
allmax = 6000;
% allmin = 500; % for inset
% allmax = 2000;

yfp_matrix1 = collage_image(namepath1,name1,cent_data1,frame,[allmin,allmax]);
yfp_matrix_rgb1 = cat(3, yfp_matrix1, yfp_matrix1, yfp_matrix1);
yfp_matrix_rgb1(:, :, 3) = 0;


yfp_matrix2 = collage_image(namepath2,name2,cent_data2,frame,[allmin,allmax]);
yfp_matrix_rgb2 = cat(3, yfp_matrix2, yfp_matrix2, yfp_matrix2);
yfp_matrix_rgb2(:, :, 3) = 0;

yfp_matrix3 = collage_image(namepath3,name3,cent_data3,frame,[allmin,allmax]);
yfp_matrix_rgb3 = cat(3, yfp_matrix3, yfp_matrix3, yfp_matrix3);
yfp_matrix_rgb3(:, :, 3) = 0;

yfp_matrix4 = collage_image(namepath4,name4,cent_data4,frame,[allmin,allmax]);
yfp_matrix_rgb4 = cat(3, yfp_matrix4, yfp_matrix4, yfp_matrix4);
yfp_matrix_rgb4(:, :, 3) = 0;

 
figure
p = tiledlayout(1,4,'TileSpacing','compact');

nexttile
k1=imresize(yfp_matrix_rgb1,[2000 1022]);
imshow(k1);
hold on
plot(flip(m1(1:10)),t,'color','#DC582A','LineWidth',2)
axis on
xticks([1 340 681 1021]);
%xticklabels({})
xticklabels({'0', '5', '10', '15'})
yticks(2000/990.*[50 149 248 347 446 545 644 743 842 941])
%yticklabels({})
yticklabels({ '7.5', '7','6.5', '6','5.5','5','4.5','4','3.5','3'})
ytickangle(0)
ylabel(p,'Time[hr]','FontSize',16)
%xlabel(p,'Distance[mm]','FontSize',24)
hold off

nexttile
k2=imresize(yfp_matrix_rgb2,[2000 1022]);
imshow(k2);
hold on
plot(flip(m2(1:10)),t,'color','#ADD8E6','LineWidth',2)
axis on
xticks([1 340 681 1021]);
%xticklabels({})
xticklabels({'0', '5', '10', '15'})
yticks(2000/990.*[50 149 248 347 446 545 644 743 842 941])
set(gca,'yticklabels',[])
hold off

nexttile
k3=imresize(yfp_matrix_rgb3,[2000 1022]);
imshow(k3);
hold on
plot(flip(m3(1:10)),t,'color','#FFC000','LineWidth',2)
axis on
xticks([1 340 681 1021]);
%xticklabels({})
xticklabels({'0', '5', '10', '15'})
yticks(2000/990.*[50 149 248 347 446 545 644 743 842 941])
set(gca,'yticklabels',[])
hold off

nexttile
k4=imresize(yfp_matrix_rgb4,[2000 1022]);
imshow(k4);
hold on
plot(flip(m4(1:10)),t,'color','#0072BD','LineWidth',2)
axis on
xticks([1 340 681 1021]);
%xticklabels({})
xticklabels({'0', '5', '10', '15'})
yticks(2000/990.*[50 149 248 347 446 545 644 743 842 941])
set(gca,'yticklabels',[])

colorMap = [linspace(0,1,256)',linspace(0,1,256)',zeros(256,1)]
colormap(colorMap);
colorbar
colorbar('XTickLabel',{'500','2000','4000','6000'}, ...
               'XTick', [0,0.27,0.64,1]);
           
hold off

%print('fig_3a.pdf','-dpdf','-fillpage')

%% Fig 3a incet

allmin = 500; % for incet
allmax = 2000;

yfp_matrix1 = collage_image(namepath1,name1,cent_data1,frame,[allmin,allmax]);
yfp_matrix_rgb1 = cat(3, yfp_matrix1, yfp_matrix1, yfp_matrix1);
yfp_matrix_rgb1(:, :, 3) = 0;
yfp_matrix_rgb1(:, :, 2) = 0;

figure

k1=imresize(yfp_matrix_rgb1,[2000 1022]);
imshow(k1);
hold on
plot(flip(m1(1:10)),t,'color','#DC582A','LineWidth',2)
axis on
xticks([1 340 681 1021]);
%xticklabels({})
xticklabels({'0', '5', '10', '15'})
yticks(2000/990.*[50 149 248 347 446 545 644 743 842 941])
%yticklabels({})
yticklabels({ '7.5', '7','6.5', '6','5.5','5','4.5','4','3.5','3'})
ytickangle(0)
ylabel('Time[hr]','FontSize',16)
%xlabel(p,'Distance[mm]','FontSize',24)

colorMap = [linspace(0,1,256)',zeros(256,2)];
colormap(colorMap);
colorbar
colorbar('XTickLabel',{'500','1000','1500','2000'}, ...
               'XTick', [0,0.33,0.67,1]);
%% Fig 3b
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

m1 = timecourse_distance(plate_1_yfp(:,ptime),0);
m2 = timecourse_distance(plate_2_yfp(:,ptime),0);
m3 = timecourse_distance(plate_3_yfp(:,ptime),0);
m21 = timecourse_distance(plate_21_yfp(:,ptime)-plate_21_yfp_bg(:,ptime),1);
m22 = timecourse_distance(plate_22_yfp(:,ptime)-plate_22_yfp_bg(:,ptime),1);
m23 = timecourse_distance(plate_23_yfp(:,ptime)-plate_23_yfp_bg(:,ptime),1);
m31 = timecourse_distance(plate_31_yfp(:,ptime),0);
m32 = timecourse_distance(plate_32_yfp(:,ptime),0);
m33 = timecourse_distance(plate_33_yfp(:,ptime),0);
m41 = timecourse_distance(plate_41_yfp(:,ptime)-plate_41_yfp_bg(:,ptime),1);
m42 = timecourse_distance(plate_42_yfp(:,ptime)-plate_42_yfp_bg(:,ptime),1);
m43 = timecourse_distance(plate_43_yfp(:,ptime)-plate_43_yfp_bg(:,ptime),1);

%%

xx1 = linspace(3,10,100);
yy1 = 1*xx1.^(1);
yy2 =3/sqrt(3)*xx1.^(1/2);

figure
box on
ptime = (1:15);
hold on
l2 = shadedErrorBar(t(ptime),[m1(ptime)*0.01468,m2(ptime)*0.01468,m3(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#DC582A'}); 
l2 = l2.mainLine;
l3= shadedErrorBar(t(ptime),[m21(ptime)*0.01468,m22(ptime)*0.01468,m23(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#ADD8E6'}); 
l3 = l3.mainLine;
l4=shadedErrorBar(t(ptime),[m31(ptime)*0.01468,m32(ptime)*0.01468,m33(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#FFC000'}); 
l4 = l4.mainLine;
l5=shadedErrorBar(t(ptime),[m41(ptime)*0.01468,m42(ptime)*0.01468,m43(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','#0072BD'}); 
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
ylabel('Wavefront-HWHM [mm]')
%ylabel('Wavefront-Thd[mm]')
 leg1=legend([l2 l3 l4 l5],'IS- PF-','IS+ PF-','IS- PF+','IS+ PF+','Location','NorthWest');
%leg1=legend([l2 l3 l4 l5 l6 l7],'QS- PF-','QS+ PF-','QS- PF+','Synthetic trigger wave','\propto t^1','\propto t^{1/2}','Location','NorthWest');
set(leg1,'FontSize',20);
ah1=axes('position',get(gca,'position'),'visible','off');
leg2=legend(ah1,[l6 l7],'\propto t^1','\propto t^{1/2}','Location','SouthEast');set(leg2,'FontSize',20);


%% fig 5 normalized new model with lag phase
% new model as the huang model described here https://www.tandfonline.com/doi/full/10.1080/13873954.2023.2236681

[num, txt,raw] = xlsread([crt_path  '/experiment data/source data file.xlsx'],'fig 4 a-b');
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

items = {'mu';'division time (min)';'y0 (log)';'ymax (log)';'lambda';'ratio (1/rho)'};
QSPF = [par_tw6_new(3);log(2)/par_tw6_new(3)*60;par_tw6_new(2);par_tw6_new(1);par_tw6_new(4);exp(par_tw6_new(1))/exp(par_tw6_new(2))];
PF = [par_pf6_new(3);log(2)/par_pf6_new(3)*60;par_pf6_new(2);par_pf6_new(1);par_pf6_new(4);exp(par_pf6_new(1))/exp(par_pf6_new(2))];
QS = [par_qs6_new(3);log(2)/par_qs6_new(3)*60;par_qs6_new(2);par_qs6_new(1);par_qs6_new(4);exp(par_qs6_new(1))/exp(par_qs6_new(2))];
NEG = [par_neg6_new(3);log(2)/par_neg6_new(3)*60;par_neg6_new(2);par_neg6_new(1);par_neg6_new(4);exp(par_neg6_new(1))/exp(par_neg6_new(2))];


T = table(QSPF,PF,QS,NEG,'RowNames',items)
%writetable(T,'fitted growthfig3_new.xlsx','WriteRowNames',true) 

ptime =[1:15];

th=0.1;

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


%% fig 4 for IS-PF- circuits

t = [3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10];

ptime = [1:15];
namepath=namepath3;% namepath1 for IS-PF-, namepath3 for IS-PF+
allcent = 1;
name_annex = '2-';
name_annex2 = '_1-15.mat';
load([namepath name_annex num2str(1) name_annex2])
plate_1_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);


load([namepath name_annex num2str(2) name_annex2])
plate_2_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_3_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

name_annex = '6-';
load([namepath name_annex num2str(1) name_annex2])
plate_21_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(2) name_annex2])
plate_22_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_23_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

name_annex = '18-';
load([namepath name_annex num2str(1) name_annex2])
plate_31_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(2) name_annex2])
plate_32_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_33_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);

%% calculate distance with half maximum
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
%ah = axes;
%l1 = plot(t(ptime),m_dye(ptime)*0.01468,'-r');
% l1 = shadedErrorBar(t(ptime),[m1_dye(ptime)*0.01468,m2_dye(ptime)*0.01468,m3_dye(ptime)*0.01468,...
%     m21_dye(ptime)*0.01468,m22_dye(ptime)*0.01468,m23_dye(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','r'}); 
% l1 = l1.mainLine;
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

%% plot growth curve for IS-PF- 
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

%% IS-PF- strain wavefront-threshold
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
title('IS- PF-, th=0.1')

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


%% fig 4 for IS+PF+ circuits

t = [3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10];

ptime = [1:15];

allcent = 1;
namepath = namepath4;% namethpath2 for IS+PF-,namethpath4 for IS+PF+
name_annex = '677-2-';
name_annex2 = '_1-15.mat';
name_annex3 = '0_';
load([namepath name_annex num2str(1) name_annex2])
plate_1_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(1) name_annex2])
plate_1_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(2) name_annex2])
plate_2_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(2) name_annex2])
plate_2_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_3_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(3) name_annex2])
plate_3_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

name_annex = '677-6-';
load([namepath name_annex num2str(1) name_annex2])
plate_21_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(1) name_annex2])
plate_21_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(2) name_annex2])
plate_22_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(2) name_annex2])
plate_22_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_23_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(3) name_annex2])
plate_23_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

name_annex = '677-18-';
load([namepath name_annex num2str(1) name_annex2])
plate_31_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(1) name_annex2])
plate_31_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(2) name_annex2])
plate_32_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
load([namepath name_annex name_annex3 num2str(2) name_annex2])
plate_32_yfp_bg = fluo_ave(YFP_data_ad,cent_data,allcent);

load([namepath name_annex num2str(3) name_annex2])
plate_33_yfp = fluo_ave(YFP_data_ad,cent_data,allcent);
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
%ah = axes;
%l1 = plot(t(ptime),m_dye(ptime)*0.01468,'-r');
% l1 = shadedErrorBar(t(ptime),[m1_dye(ptime)*0.01468,m2_dye(ptime)*0.01468,m3_dye(ptime)*0.01468,...
%     m21_dye(ptime)*0.01468,m22_dye(ptime)*0.01468,m23_dye(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','r'}); 
% l1 = l1.mainLine;
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


%% plot growth curve for IS+PF+
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

%% IS+PF+ wavefront-threshold
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
%ah = axes;
%l1 = plot(t(ptime),m_dye(ptime)*0.01468,'-r');
% l1 = shadedErrorBar(t(ptime),[m1_dye(ptime)*0.01468,m2_dye(ptime)*0.01468,m3_dye(ptime)*0.01468,...
%     m21_dye(ptime)*0.01468,m22_dye(ptime)*0.01468,m23_dye(ptime)*0.01468]',{@mean,@std},'lineprops',{'-o','color','r'}); 
% l1 = l1.mainLine;
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
title('IS+ PF+, th=0.1')

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

