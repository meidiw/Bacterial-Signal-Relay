clear all

tic
D = 500/(1e3)^2*3600; % mm^2/h
s0 = 2000; % ug/ml
R0 = .5; % mm
alpha = 4; % 1/h

% control params
lambda1 = 0.616418776; % h
rho1 = 145.3173138/38759.49764;
g1 = 1.534823959; % 1/h
H1 = 1.6;
s1 = 8.7; % ug/ml

% +PF params
lambda2 = 1.442962528; % h
rho2 = 219.657811/25863.81969;
g2 = 1.522073485; % 1/h
J2 = 2; % sets sharpness of dip
Z2 = 0.5; % sets time of tip
b2 = 3.5; % when large, makes dip pronounced

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
r = linspace(0,R,1000)';
M = length(r);

T = 10; % h, determined by experiments 
t = linspace(0.01,T,1000)'; 
N = length(t);


m = 1;
opts = odeset('AbsTol',1e-3,'RelTol',1e-3);


% quorum sensing
solqs = pdepe(m,@(r,t,u,dudx)qs_pde(r,t,u,dudx,D,s0,H3,s3,a3,alpha,lambda3,rho3,g3),...
    @step_ic,@step_bc,r,t,opts);
vqs = solqs(:,:,2)';

vqsy = solqs(:,:,3)';

% control
solcontrol = pdepe(m,@(r,t,u,dudx)control_pde(r,t,u,dudx,D,s0,H1,s1,alpha,lambda1,rho1,g1),...
    @step_ic,@step_bc,r,t,opts);
vcontrol = solcontrol(:,:,2)';

vcontroly = solcontrol(:,:,3)';

% positive feedback
solfb = pdepe(m,@(r,t,u,dudx)fb_pde(r,t,u,dudx,D,s0,s1,H1,J2,Z2,b2,alpha,lambda2,rho2,g2),...
    @step_ic,@step_bc,r,t,opts);
vfb = solfb(:,:,2)';

vfby = solfb(:,:,3)';

% positive feedback + quorum sensing (trigger wave)
solfbqs = pdepe(m,@(r,t,u,dudx)fbqs_pde(r,t,u,dudx,D,s0,H3,J4,s3,a3,b4,Z4,alpha,lambda4,rho4,g4),...
    @step_ic,@step_bc,r,t,opts);
vfbqs = solfbqs(:,:,2)';

vfbqsy = solfbqs(:,:,3)';

% half-max distance - quorum sensing
vnormqs = vqsy./(ones(length(r),1)*vqsy(1,:));
[~,i] = min(abs(vnormqs - .5),[],1);
chistarqs = r(i);

% half-max distance - control
vcontrolnorm = vcontroly./(ones(length(r),1)*vcontroly(1,:));
[~,i] = min(abs(vcontrolnorm - .5),[],1);
chistarcontrol = r(i);

% half-max distance - positive feedback
vfbnorm = vfby./(ones(length(r),1)*vfby(1,:));
[~,i] = min(abs(vfbnorm - .5),[],1);
chistarfb = r(i);

% half-max distance - trigger wave
vnormfbqs = vfbqsy./(ones(length(r),1)*vfbqsy(1,:));
[~,i] = min(abs(vnormfbqs - .5),[],1);
chistarfbqs = r(i);



% plotting

[~,indmin] = min(abs(t-3));
[~,indmax] = min(abs(t-7.5));


[~,idx] = min(abs(t-7)); % to make maximum intensity appear around t = 7.5 h

% creating custom colormap to match experiment

yellowMap = [linspace(0, 1, 256)', linspace(0, 1, 256)', ones(256, 1).*(8/256)];

colormap(yellowMap);

g = [1.909375714,1.522073485,2.362465926,1.534823959];
lambda = sqrt(D./g) * 10^-3; % mm


fig = figure(1);clf
fig.Position = [25,25,20,10];
fig.Units = 'inches';
tiledlayout(fig,"horizontal");
p = tiledlayout(1,4,'TileSpacing','compact');
set(groot, 'defaultLineLineWidth', 2.0);
% subplot(1,4,1)
nexttile;
h1 = pcolor(r,t,vcontroly'/max(vcontroly(:,idx)));
% clim([0 max(vcontroly(:,end))])
clim([0 1])
set(h1, 'EdgeColor', 'none');
hold on
plot(chistarcontrol(indmin:indmax),t(indmin:indmax),'Color','#DC582A','LineWidth', 3)
hold off
xlim([0 15])
ylim([2.5 8])
set(gca, 'YTick', 3:0.5:7.5);
title("IS- PF-")
% colorbar


% subplot(1,4,2)
nexttile
h3 = pcolor(r,t,vqsy'/max(vqsy(:,idx)));

% colorbar
% clim([0 max(vqsy(:,end))])
clim([0 1])
set(gca, 'YTick', 3:0.5:7.5);
set(gca,'YTickLabel',[]);
set(h3, 'EdgeColor', 'none');

% plotting hwhm
hold on
plot(chistarqs(indmin:indmax),t(indmin:indmax),'Color','#ADD8E6','LineWidth', 3)
hold off
xlim([0 15])
ylim([2.5 8])
set(gca, 'YTick', 3:0.5:7.5);
title("IS+ PF-")


% subplot(1,4,3)
nexttile
h2 = pcolor(r,t,vfby'/max(vfby(:,idx)));


% colorbar
% clim([0 max(vfby(:,end))])
clim([0 1])
set(gca, 'YTick', 3:0.5:7.5);
set(gca,'YTickLabel',[]);
set(h2, 'EdgeColor', 'none');

% plotting hwhm
hold on
plot(chistarfb(indmin:indmax),t(indmin:indmax),'Color','#FFB38F','LineWidth', 3)
hold off
xlim([0 15])
ylim([2.5 8])
set(gca, 'YTick', 3:0.5:7.5);
title("IS-- PF+")


% subplot(1,4,4)
nexttile
h4 = pcolor(r,t,vfbqsy'/max(vfbqsy(:,idx)));

% colorbar
% clim([0 max(vfbqsy(:,end))])
clim([0 1])
set(gca, 'YTick', 3:0.5:7.5);
set(gca,'YTickLabel',[]);
set(h4, 'EdgeColor', 'none');

% plotting hwhm
hold on
plot(chistarfbqs(indmin:indmax),t(indmin:indmax),'Color','#0072BD','LineWidth', 3)
hold off
xlim([0 15])
ylim([2.5 8])
set(gca, 'YTick', 3:0.5:7.5);
title("IS+ PF+")


han=axes(fig,'visible','off'); 
han.XLabel.Visible='on';
han.YLabel.Visible='on';
fontsize(20,"points")
ylabel(p,'Time [hr]','FontSize',24);
xlabel(p,'Distance [mm]','FontSize',24);
box on


lw = 1.5;
zmax = 3;
ymax = 1.2;
rmax = 30;

toc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [c,f,s] = qs_pde(r,t,u,dudx,D,s0,H3,s3,a3,alpha,lambda3,rho3,g3)
c = [1; 1; 1];
E = 1.6 * D;
f = [D * dudx(1); E * dudx(2); 0];

B3 = t + log((1+exp(-alpha*(t-lambda3)))./(1+exp(alpha*lambda3)))/alpha;
phi3 = rho3./(rho3+exp(-g3*B3));
B3dot = 1./(1+exp(-alpha*(t-lambda3)));
f3 = rho3^(3/2)*exp(-g3*B3/2).*sqrt(B3dot)./(rho3+exp(-g3*B3)).^2;

% f3 = (rho3 * exp(g3 * B3) .* sqrt(B3dot))./(rho3 * exp(g3*B3) + 1).^(3/2);

z3 = (u(1).^H3) ./ (u(1).^H3 + s3.^H3);
s = [0; f3 .* a3 * z3; f3 .* u(2)];
% s = [0; f3 .* a3 * z3; f3 .* y3];

end

function [c,f,s] = control_pde(r,t,u,dudx,D,s0,H1,s1,alpha,lambda1,rho1,g1)
c = [1; 1; 1];
f = [D; 0; 0] .* dudx(1); %  makes pde into ode w/ diff. source, check docs

B1 = t + log((1+exp(-alpha*(t-lambda1)))/(1+exp(alpha*lambda1)))/alpha;
phi1 = rho1./(rho1+exp(-g1*B1));
B1dot = 1./(1+exp(-alpha*(t-lambda1)));
f1 = rho1^(3/2)*exp(-g1*B1/2).*sqrt(B1dot)./(rho1+exp(-g1*B1)).^2;
rbar_ = f1 .* ( (u(1).^H1)./(u(1).^H1 + s1^H1) );

s = [0; rbar_; f1 .* u(2)];
end
 
function [c,f,s] = fb_pde(r,t,u,dudx,D,s0,s1,H1,J2,Z2,b2,alpha,lambda2,rho2,g2)
c = [1; 1; 1];
f = [D; 0; 0] .* dudx(1); % makes pde into ode w/ diff. source, check docs

B2 = t + log((1+exp(-alpha*(t-lambda2)))/(1+exp(alpha*lambda2)))/alpha;
phi2 = rho2./(rho2+exp(-g2*B2));
B2dot = 1./(1+exp(-alpha*(t-lambda2)));
f2 = rho2^(3/2)*exp(-g2*B2/2).*sqrt(B2dot)./(rho2+exp(-g2*B2)).^2;
rbar_ = f2 .* ( (u(1).^H1)./(u(1).^H1 + s1.^H1)  + b2 * ( (u(2).^J2)./(u(2).^J2 + Z2.^J2) ) ) ;

s = [0; rbar_; f2 .* u(2)];
end


function [c,f,s] = fbqs_pde(r,t,u,dudx,D,s0,H3,J4,s3,a3,b4,Z4,alpha,lambda4,rho4,g4)
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
R0 = 0.5; % mm 
s0 = 2000; % ug/ml
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


