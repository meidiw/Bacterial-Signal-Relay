clear all

tic
D = 500/(1e3)^2*3600; % mm^2/h
R0 = .5; % mm
alpha = 4; % 1/h


% +QS params
lambda3 = 1.07863189; % h
rho3 = 109.5097541/43240.16717;
g3 = 2.362465926; % 1/h
H3 = 1.9;
s3 = 6.3; % ug/ml
a3 = 1;

% +PF+QS params
lambda4 = 0.99316237; % h
rho4 = 235.09936/41724.74796;
g4 = 1.909375714; % 1/h
J4 = 2; 
Z4 = 0.05; 
b4 = 3.5; 


R = 30; % mm
r = linspace(0,30,1500)';
M = length(r);

T = 10; % h, determined by experiments 
t = linspace(0.01,10,1000)'; 
N = length(t);


m = 1;
opts = [];

s0list = [1000, 2000, 3000];

beta = 1/2;

color = {'#BBE6B3','#7EC77D','#3D8E4F'};


% Initialize the figure
f1 = figure(1); clf;

lw = 2;
fs = 20;

set(groot, 'defaultLineLineWidth', 2);

set(f1,'DefaultTextFontsize',15, ...
 'DefaultTextFontname','Arial', ...
 'DefaultTextFontWeight','bold', ...
 'DefaultAxesFontsize',15, ...
 'DefaultAxesFontname','Arial', ...
 'DefaultLineLineWidth', 2)


for ii = 1:3

    k = s0list(ii);

    global s0;

    s0 = k;

    % positive feedback + quorum sensing (trigger wave)

    solfbqs = pdepe(m,@(r,t,u,dudx)fbqs_pde(r,t,u,dudx,D,s0,H3,J4,s3,a3,b4,Z4,alpha,lambda4,rho4,g4,beta),...
        @step_ic,@step_bc,r,t,opts);
    vfbqs = solfbqs(:,:,2)';

    vfbqsy = solfbqs(:,:,3)';

    % half-max distance - trigger wave
    vnormfbqs = vfbqsy./(ones(length(r),1)*vfbqsy(1,:));
    [~,i] = min(abs(vnormfbqs - .5),[],1);
    chistarfbqs = r(i);
    loglog(t,chistarfbqs,color=color{ii})
    hold on


end


% Wavefronts data
t_ = [3 10]; % h
whalf = 3 * (t_ / 3).^.5;
wone = 3 * (t_ / 3);

ydiffplot = loglog(t_,whalf,':','Color','#808080',LineWidth=2);
hold on
ybalisticplot = loglog(t_,wone,'k:',LineWidth=2);
xlim([3 10])
ylim([2 40])
yticks([2 4 6 8 10]);
yticklabels({'2', '4', '6','8', '10'})
xlabel('Time [hr]','FontSize',20)
ylabel('HWHM [mm]','FontSize',20)
lgdlist = [2,4,6];
title("IS+ PF+, \beta = " + beta); legendStrings = "eryC = " + string(lgdlist) + " ${\rm \mu g}$"; 
lgd = legend(legendStrings, 'Location','Northwest','Interpreter','latex');
fontsize(lgd,20,'points')
% legend(,'Location','nw')
ax = gca;
ax.LineWidth = 2;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;
ax.Title.FontSize = 20;
curtick = get(ax, 'xTick');
xticks(unique(round(curtick)));

legend boxoff

% 2nd legend
% invisible dummy axes object for the second legend
Ax2=axes('Position',get(gca,'Position'),'Visible','Off');
legend(Ax2,[ybalisticplot ydiffplot],{'\propto t^{1}','\propto t^{1/2}'},...
    'Location','Southeast','LineWidth',2.0,'FontSize',20);

legend boxoff

toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [c,f,s] = fbqs_pde(r,t,u,dudx,D,s0,H3,J4,s3,a3,b4,Z4,alpha,lambda4,rho4,g4,beta)
E = 1.6 * D;
c = [1; 1; 1];
f = [D * dudx(1); E * dudx(2); 0];

B4 = t + log((1+exp(-alpha*(t-lambda4)))/(1+exp(alpha*lambda4)))/alpha;
phi4 = rho4./(rho4+exp(-g4*B4));
B4dot = 1./(1+exp(-alpha*(t-lambda4)));
f4 = rho4^(3/2)*exp(-g4*B4/2).*sqrt(B4dot)./(rho4+exp(-g4*B4)).^2;

s = [0; f4 .* ( a3*(u(1).^H3)./(u(1).^H3 + s3.^H3) + b4*((u(2).^J4)./(u(2).^J4 + Z4^J4)) ); ...
     f4 .* u(2)];
end

function v0 = step_ic(r)
global s0;
R0 = 0.5; % mm
% s0 = 2000; % ug/ml
    if r < R0
        v0 = [s0; 0; 0];
    else
        v0 = [0; 0; 0];
    end
end



function [pl,ql,pr,qr] = step_bc(rl,vl,rr,vr,t)
pl = [0; 0; 0];
ql = [1; 1; 1]; % reflecting at r = 0
pr = [vr(1); vr(2); vr(3)];
qr = [0; 0; 0]; % absorbing at r = chimax
end

