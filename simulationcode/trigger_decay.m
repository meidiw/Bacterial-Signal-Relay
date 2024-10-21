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
Z4 = 0.03; 
b4 = 3.5; 

R = 30; % mm
% r = [0 logspace(-2,log10(R),1e3)]';%logspace(-3,3,1e3)';
% r = logspace(-2,log10(R),1e4)';%logspace(-3,3,1e3)';
r = linspace(0,30,2500)';
M = length(r);

T = 10; % h, determined by experiments 
% t = [0 logspace(-3,log10(T),1e2)]';%logspace(-3,2,1e2);
% t = logspace(-3,log10(T),1e2)';%logspace(-3,2,1e2);
t = linspace(0,10,1500)'; 
N = length(t);


m = 1;
opts = odeset('AbsTol',1e-3,'RelTol',1e-3);


% positive feedback, default values
Z2default = 0.5;
mudefault = 0;
solfb = pdepe(m,@(r,t,u,dudx)fb_ode(r,t,u,dudx,D,s0,s1,H1,J2,Z2default,b2,alpha,lambda2,rho2,g2,mudefault),...
    @step_ic,@step_bc,r,t,opts);

vfby1 = solfb(:,:,3)';

% half-max distance - positive feedback
vfbnorm1 = vfby1./(ones(length(r),1)*vfby1(1,:));
[~,i] = min(abs(vfbnorm1 - .5),[],1);
chistarfbdefault = r(i);

% positive feedback, z2 == z4
Z2equal = 0.05;
muequal = 0;
solfb = pdepe(m,@(r,t,u,dudx)fb_ode(r,t,u,dudx,D,s0,s1,H1,J2,Z2equal,b2,alpha,lambda2,rho2,g2,muequal),...
    @step_ic,@step_bc,r,t,opts);

vfby2 = solfb(:,:,3)';

% half-max distance - positive feedback
vfbnorm2 = vfby2./(ones(length(r),1)*vfby2(1,:));
[~,i] = min(abs(vfbnorm2 - .5),[],1);
chistarfbz2z4 = r(i);

% positive feedback, z2 == z4 and mu = 10
Z2equalmu = 0.05;
muequalmu = 12;
solfb = pdepe(m,@(r,t,u,dudx)fb_ode(r,t,u,dudx,D,s0,s1,H1,J2,Z2equalmu,b2,alpha,lambda2,rho2,g2,muequalmu),...
    @step_ic,@step_bc,r,t,opts);

vfby3 = solfb(:,:,3)';

% half-max distance - positive feedback
vfbnorm3 = vfby3./(ones(length(r),1)*vfby3(1,:));
[~,i] = min(abs(vfbnorm3 - .5),[],1);
chistarfbz2z4mu = r(i);


% Initialize the figure
f1 = figure(1); clf;

lw = 4;
fs = 20;

set(groot, 'defaultLineLineWidth', 2);

set(f1,'DefaultTextFontsize',15, ...
 'DefaultTextFontname','Arial', ...
 'DefaultTextFontWeight','bold', ...
 'DefaultAxesFontsize',15, ...
 'DefaultAxesFontname','Arial', ...
 'DefaultLineLineWidth', 2)

firstax = axes(f1, 'FontSize', 20); 
hold(firstax, 'on'); % Ensure hold is applied to firstax

% Wavefronts data
t_ = [3 10]; % h
whalf = 3 * (t_ / 3).^.5;
wone = 3 * (t_ / 3);

% Plot data
h0 = loglog(firstax, t, chistarfbdefault, t, chistarfbz2z4, t, chistarfbz2z4mu,'LineWidth', 2);
xlim(firstax, [3 10]);
ylim(firstax, [2 15]);
xlabel(firstax, 'Time [hr]', 'FontSize', fs);
ylabel(firstax, 'HWHM [mm]','FontSize',fs);
set(firstax, 'Box', 'on');
set(firstax, 'YTick', 2:2:10);

% Scale lines
ydiffplot = loglog(t_,whalf,':','Color','#808080',LineWidth=2);
hold on
ybalisticplot = loglog(t_,wone,'k:',LineWidth=2);


% First legend
leg1 = legend(firstax, h0, {'IS-PF+, $$z_2 = 0.3 \, {\rm \mu g/mL}$$', ...
    'IS-PF+, $$z_2 = 0.03 \, {\rm \mu g/mL}$$', ...
    'IS-PF+, $$z_2 = 0.03 \, {\rm \mu g/mL}$$, w/ degradation'}, ...
    'Location', 'northwest','Interpreter','latex');
set(leg1, 'FontSize', 20,'LineWidth',2);

legend boxoff

% 2nd legend
% invisible dummy axes object for the second legend
Ax2=axes('Position',get(gca,'Position'),'Visible','Off');
legend(Ax2,[ybalisticplot ydiffplot],{'\propto t^{1}','\propto t^{1/2}'},...
    'Location','Southeast','LineWidth',2.0,'FontSize',20);

legend boxoff


dir = '../fig/simulation/';

saveas(f1,[ dir 'suppfigure_decay'],'pdf')

toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [c,f,s] = qs_pde(r,t,u,dudx,D,s0,H3,s3,a3,alpha,lambda3,rho3,g3)
c = [1; 1; 1];
f = D .* [dudx(1); dudx(2); 0];

B3 = t + log((1+exp(-alpha*(t-lambda3)))./(1+exp(alpha*lambda3)))/alpha;
phi3 = rho3./(rho3+exp(-g3*B3));
B3dot = 1./(1+exp(-alpha*(t-lambda3)));
% f3 = rho3^(3/2)*exp(-g3*B3/2).*sqrt(B3dot)./(rho3+exp(-g3*B3)).^2;

f3 = (rho3 * exp(g3 * B3) .* sqrt(B3dot))./(rho3 * exp(g3*B3) + 1).^(3/2);

z3 = (u(1).^H3) ./ (u(1).^H3 + s3.^H3);
s = [0; f3 .* a3 * z3; f3 .* u(2)];

end

function [c,f,s] = control_ode(r,t,u,dudx,D,s0,H1,s1,alpha,lambda1,rho1,g1)
c = [1; 1; 1];
f = [D; 0; 0] .* dudx(1); %  makes pde into ode w/ diff. source, check docs

B1 = t + log((1+exp(-alpha*(t-lambda1)))/(1+exp(alpha*lambda1)))/alpha;
phi1 = rho1./(rho1+exp(-g1*B1));
B1dot = 1./(1+exp(-alpha*(t-lambda1)));
f1 = rho1^(3/2)*exp(-g1*B1).*sqrt(B1dot)./(rho1+exp(-g1*B1)).^2;

f1 = (rho1 * exp(g1 * B1) .* sqrt(B1dot))./(rho1 * exp(g1*B1) + 1).^(3/2);


rbar_ = f1 .* ( (u(1).^H1)./(u(1).^H1 + s1^H1) );
s = [0; rbar_; f1 .* u(2)];
end
 
function [c,f,s] = fb_ode(r,t,u,dudx,D,s0,s1,H1,J2,Z2,b2,alpha,lambda2,rho2,g2,mu)
c = [1; 1; 1];
f = [D; 0; 0] .* dudx(1); % makes pde into ode w/ diff. source, check docs

B2 = t + log((1+exp(-alpha*(t-lambda2)))/(1+exp(alpha*lambda2)))/alpha;
phi2 = rho2./(rho2+exp(-g2*B2));
B2dot = 1./(1+exp(-alpha*(t-lambda2)));
f2 = (rho2 * exp(g2 * B2) .* sqrt(B2dot))./(rho2 * exp(g2*B2) + 1).^(3/2);

rbar_ = f2 .* ( (u(1).^H1)./(u(1).^H1 + s1.^H1)  + b2 * ( (u(2).^J2)./(u(2).^J2 + Z2.^J2) ) ) - mu*u(2);
s = [0; rbar_; f2 .* u(2)];
end


function [c,f,s] = fbqs_pde(r,t,u,dudx,D,s0,H3,J4,s3,a3,b4,Z4,alpha,lambda4,rho4,g4)
c = [1; 1; 1];
f = D .* [dudx(1); dudx(2); 0];

B4 = t + log((1+exp(-alpha*(t-lambda4)))/(1+exp(alpha*lambda4)))/alpha;
phi4 = rho4./(rho4+exp(-g4*B4));
B4dot = 1./(1+exp(-alpha*(t-lambda4)));
f4 = rho4^(3/2)*exp(-g4*B4/2).*sqrt(B4dot)./(rho4+exp(-g4*B4)).^2;

f4 = (rho4 * exp(g4 * B4) .* sqrt(B4dot))./(rho4 * exp(g4*B4) + 1).^(3/2);

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