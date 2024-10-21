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
% r = [0 logspace(-2,log10(R),1e3)]';%logspace(-3,3,1e3)';
% r = logspace(-2,log10(R),1e4)';%logspace(-3,3,1e3)';
r = linspace(0,30,2000)';
M = length(r);

T = 10; % h, determined by experiments 
% t = [0 logspace(-3,log10(T),1e2)]';%logspace(-3,2,1e2);
% t = logspace(-3,log10(T),1e2)';%logspace(-3,2,1e2);
t = linspace(0,10,1500)'; 
N = length(t);



m = 1;
opts = odeset('AbsTol',1e-3,'RelTol',1e-3);


% growth functions
B1 = t + log((1+exp(-alpha*(t-lambda1)))/(1+exp(alpha*lambda1)))/alpha;
phi1 = rho1./(rho1+exp(-g1*B1));
B1dot = 1./(1+exp(-alpha*(t-lambda1)));

B2 = t + log((1+exp(-alpha*(t-lambda2)))/(1+exp(alpha*lambda2)))/alpha;
phi2 = rho2./(rho2+exp(-g2*B2));
B2dot = 1./(1+exp(-alpha*(t-lambda2)));

B3 = t + log((1+exp(-alpha*(t-lambda3)))/(1+exp(alpha*lambda3)))/alpha;
phi3 = rho3./(rho3+exp(-g3*B3));
B3dot = 1./(1+exp(-alpha*(t-lambda3)));

B4 = t + log((1+exp(-alpha*(t-lambda4)))/(1+exp(alpha*lambda4)))/alpha;
phi4 = rho4./(rho4+exp(-g4*B4));
B4dot = 1./(1+exp(-alpha*(t-lambda4)));




% quorum sensing
solqs = pdepe(m,@(r,t,u,dudx)qs_pde(r,t,u,dudx,D,s0,H3,s3,a3,alpha,lambda3,rho3,g3),...
    @step_ic,@step_bc,r,t,opts);
% vqs = solqs(:,:,2)';

vqsy = solqs(:,:,3);


% control
solcontrol = pdepe(m,@(r,t,u,dudx)control_pde(r,t,u,dudx,D,s0,H1,s1,alpha,lambda1,rho1,g1),...
    @step_ic,@step_bc,r,t,opts);
% vcontrol = solcontrol(:,:,2)';

vcontroly = solcontrol(:,:,3);


% positive feedback
solfb = pdepe(m,@(r,t,u,dudx)fb_pde(r,t,u,dudx,D,s0,s1,H1,J2,Z2,b2,alpha,lambda2,rho2,g2),...
    @step_ic,@step_bc,r,t,opts);
% vfb = solfb(:,:,2)';

vfby = solfb(:,:,3);


% positive feedback + quorum sensing (trigger wave)
solfbqs = pdepe(m,@(r,t,u,dudx)fbqs_pde(r,t,u,dudx,D,s0,H3,J4,s3,a3,b4,Z4,alpha,lambda4,rho4,g4),...
    @step_ic,@step_bc,r,t,opts);
% vfbqs = solfbqs(:,:,2)';

vfbqsy = solfbqs(:,:,3);


thresh = 0.001;

phi1mat = ones(M,1)*phi1';
y1phi = vcontroly./phi1mat';

phi2mat = ones(M,1)*phi2';
y2phi = vfby./phi2mat';

phi3mat = ones(M,1)*phi3';
y3phi = vqsy./phi3mat';

phi4mat = ones(M,1)*phi4';
y4phi = vfbqsy./phi4mat';

w1th = zeros(N,1);
w2th = zeros(N,1);
w3th = zeros(N,1);
w4th = zeros(N,1);

for ii = 1:N
    [~,jth1] = min(abs(y1phi(ii,:)-thresh*max(y1phi(end,:))));
    w1th(ii) = r(jth1);

    [~,jth2] = min(abs(y2phi(ii,:)-thresh*max(y2phi(end,:))));
    w2th(ii) = r(jth2);
    
    [~,jth3] = min(abs(y3phi(ii,:)-thresh*max(y3phi(end,:))));
    w3th(ii) = r(jth3);
     
    [~,jth4] = min(abs(y4phi(ii,:)-thresh*max(y4phi(end,:))));
    w4th(ii) = r(jth4);
end


% fbqs - threshold method
% vfbqsyn = vfbqsy./nfbqs;
% [~,i] = min(abs(vfbqsyn - thresh*max(vfbqsyn,[],"all")),[],1);
% chistar_fbqs = r(i);
% 
% qs - threshold method
% vqsyn = vqsy./nqs;
% [~,i] = min(abs(vqsyn - thresh*max(vqsyn,[],"all")),[],1);
% chistar_qs = r(i);
% 
% fb - threshold method
% vfbyn = vfby./nfb;
% [~,i] = min(abs(vfbyn - thresh*max(vfbyn,[],"all")),[],1);
% chistar_fb = r(i);
% 
% control - threshold method
% vcontrolyn = vcontroly./ncontrol;
% [~,i] = min(abs(vcontrolyn - thresh*max(vcontrolyn,[],"all")),[],1);
% chistarcontrol = r(i);

% yaxisplots = [chistarfb,chistarfbqs, ...
%     chistarqs,chistarcontrol];

[~,ind] = min(abs(t-3.5));
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


firstax = axes(f1, 'FontSize', fs); 
hold(firstax, 'on'); % Ensure hold is applied to firstax


% Wavefronts data
t_ = [3 10]; % h
whalf = 3 * (t_ / 3).^.5;
wone = 3 * (t_ / 3);


% Plot data
h0 = loglog(firstax, t(ind:end),w1th(ind:end), t(ind:end),w3th(ind:end), ...
    t(ind:end),w2th(ind:end), t(ind:end),w4th(ind:end), 'LineWidth', lw);
h0(1).Color = '#DC582A';
h0(2).Color = '#ADD8E6';
h0(3).Color = '#FFC000';
h0(4).Color = '#0072BD';
xlim(firstax, [3 10]);
ylim(firstax, [2 60]);
xlabel(firstax, 'Time [hr]', 'FontSize', fs);
ylabel(firstax, 'Threshold [mm]','FontSize',fs);
title("k = " + 0.5)
set(firstax, 'Box', 'on');
set(firstax, 'YTick', 2:2:10);

% Add the additional lines
h1 = loglog(firstax, t_, wone, 'k:','LineWidth',lw);
h2 = loglog(firstax, t_, whalf, ':','Color', '#808080', 'LineWidth', lw);

% First legend
leg1 = legend(firstax, h0, {'IS- PF-', 'IS+ PF-', 'IS- PF+', 'IS+ PF+'}, ...
    'Location', 'northwest');
set(leg1, 'FontSize', fs,'LineWidth',lw);

legend boxoff

% Second axes for additional legend
ah1 = axes('Position', get(firstax, 'Position'), 'Visible', 'off');
leg2 = legend(ah1, [h1, h2], {'\propto t^{1}', '\propto t^{1/2}'}, 'Location', 'southeast');
set(leg2, 'FontSize', fs,'LineWidth',lw);
set(firstax,'XScale','log','YScale','log','LineWidth',lw)

legend boxoff

% dir = '../fig/simulation/';
% 
% saveas(f1,[ dir 'figure5_b'],'pdf')

toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% function [c,f,s] = qs_pde(r,t,u,dudx,D,s0,H3,s3,a3,alpha,lambda3,rho3,g3)
% E = 1.6 * D;
% c = [1; 1; 1];
% f = [D * dudx(1); E * dudx(2); 0];
% 
% B3 = t + log((1+exp(-alpha*(t-lambda3)))./(1+exp(alpha*lambda3)))/alpha;
% phi3 = rho3./(rho3+exp(-g3*B3));
% B3dot = 1./(1+exp(-alpha*(t-lambda3)));
% 
% f3 = (rho3^2 * exp(2 * g3 * B3) .* B3dot).^(1/2)./(rho3 * exp(g3*B3) + 1).^(3/2);
% 
% % f3 = (rho3^2 * exp(2 * g3 * B3) .* B3dot)^(1/4)./(rho3 * exp(g3*B3) + 1).^(2/4);
% 
% z3 = (u(1).^H3) ./ (u(1).^H3 + s3.^H3);
% s = [0; f3 .* a3 * z3; f3 .* u(2)];
% 
% end
% 
% function [c,f,s] = control_pde(r,t,u,dudx,D,s0,H1,s1,alpha,lambda1,rho1,g1)
% E = 1.6 * D;
% c = [1; 1; 1];
% f = [D; 0; 0] .* dudx(1); %  makes pde into ode w/ diff. source, check docs
% 
% B1 = t + log((1+exp(-alpha*(t-lambda1)))/(1+exp(alpha*lambda1)))/alpha;
% phi1 = rho1./(rho1+exp(-g1*B1));
% B1dot = 1./(1+exp(-alpha*(t-lambda1)));
% 
% f1 = (rho1^2 * exp(2 * g1 * B1) .* B1dot).^(1/2)./(rho1 * exp(g1*B1) + 1).^(3/2);
% 
% % f1 = (rho1^2 * exp(2 * g1 * B1) .* B1dot)^(1/4)./(rho1 * exp(g1*B1) + 1).^(2/4);
% 
% rbar_ = f1 .* ( (u(1).^H1)./(u(1).^H1 + s1^H1) );
% s = [0; rbar_; f1 .* u(2)];
% end
% 
% function [c,f,s] = fb_pde(r,t,u,dudx,D,s0,s1,H1,J2,Z2,b2,alpha,lambda2,rho2,g2)
% E = 1.6 * D;
% c = [1; 1; 1];
% f = [D; 0; 0] .* dudx(1); % makes pde into ode w/ diff. source, check docs
% 
% B2 = t + log((1+exp(-alpha*(t-lambda2)))/(1+exp(alpha*lambda2)))/alpha;
% phi2 = rho2./(rho2+exp(-g2*B2));
% B2dot = 1./(1+exp(-alpha*(t-lambda2)));
% 
% f2 = (rho2^2 * exp(2 * g2 * B2) .* B2dot).^(1/2)./(rho2 * exp(g2*B2) + 1).^(3/2);
% 
% % f2 = (rho2^2 * exp(2 * g2 * B2) .* B2dot).^(1/4)./(rho2 * exp(g2*B2) + 1).^(2/4);
% 
% rbar_ = f2 .* ( (u(1).^H1)./(u(1).^H1 + s1.^H1)  + b2 * ( (u(2).^J2)./(u(2).^J2 + Z2.^J2) ) ) ;
% s = [0; rbar_; f2 .* u(2)];
% end
% 
% 
% function [c,f,s] = fbqs_pde(r,t,u,dudx,D,s0,H3,J4,s3,a3,b4,Z4,alpha,lambda4,rho4,g4)
% E = 1.6 * D;
% c = [1; 1; 1];
% f = [D * dudx(1); E * dudx(2); 0];
% 
% B4 = t + log((1+exp(-alpha*(t-lambda4)))/(1+exp(alpha*lambda4)))/alpha;
% phi4 = rho4./(rho4+exp(-g4*B4));
% B4dot = 1./(1+exp(-alpha*(t-lambda4)));
% 
% f4 = (rho4^2 * exp(2 * g4 * B4) .* B4dot).^(1/2)./(rho4 * exp(g4*B4) + 1).^(3/2);
% 
% f4 = ( (rho4 * exp(g4*B4) .* B4dot)./(rho4 * exp(g4*B4) + 1).^2 )^(1/3) ...
%     .* ( ( rho4 * exp(g4*B4) ) ./( rho4*exp(g4*B4) + 1 ) )^(1-(1/3));
% 
% % f4 = (rho4^2 * exp(2 * g4 * B4) .* B4dot).^(1/4)./(rho4 * exp(g4*B4) + 1).^(2/4);
% 
% 
% s = [0; f4 .* ( a3*(u(1).^H3)./(u(1).^H3 + s3.^H3) + b4*((u(2).^J4)./(u(2).^J4 + Z4^J4)) ); ...
%      f4 .* u(2)];
% end
% 
% 
% function v0 = step_ic(r)
% R0 = 0.5; % mm 
% s0 = 2000; % ug/ml
%     if r < R0
%         v0 = [s0; 0; 0];
%     else
%         v0 = [0; 0; 0];
%     end
% end
% 
% 
% function [pl,ql,pr,qr] = step_bc(rl,vl,rr,vr,t)
% pl = [0; 0; 0];
% ql = [1; 1; 1]; % reflecting at r = 0
% pr = [vr(1); vr(2); vr(3)];
% qr = [0; 0; 0]; % absorbing at r = chimax
% end

function [c,f,s] = qs_pde(r,t,u,dudx,D,s0,H3,s3,a3,alpha,lambda3,rho3,g3)
c = [1; 1; 1];
f = D .* [dudx(1); dudx(2); 0];

B3 = t + log((1+exp(-alpha*(t-lambda3)))./(1+exp(alpha*lambda3)))/alpha;
phi3 = rho3./(rho3+exp(-g3*B3));
B3dot = 1./(1+exp(-alpha*(t-lambda3)));
f3 = rho3^(3/2)*exp(-g3*B3/2).*sqrt(B3dot)./(rho3+exp(-g3*B3)).^2;

f3 = (rho3^2 * exp(2*g3*B3) .* B3dot)^(1/6)./(rho3*exp(g3*B3)+1).^(3/6);

f3 = (rho3 * exp(g3 * B3) .* sqrt(B3dot))./(rho3 * exp(g3*B3) + 1).^(3/2);
k = 0.5;

z3 = (u(1).^H3) ./ (u(1).^H3 + s3.^H3);
y3 = (u(2)^2) ./ (u(2)^2 + k^2);
s = [0; f3 .* a3 * z3; f3 .* (u(2)^1) ./ (u(2)^1 + k^1)];
% s = [0; f3 .* a3 * z3; f3 .* y3];

end

function [c,f,s] = control_pde(r,t,u,dudx,D,s0,H1,s1,alpha,lambda1,rho1,g1)
c = [1; 1; 1];
f = [D; 0; 0] .* dudx(1); %  makes pde into ode w/ diff. source, check docs

B1 = t + log((1+exp(-alpha*(t-lambda1)))/(1+exp(alpha*lambda1)))/alpha;
phi1 = rho1./(rho1+exp(-g1*B1));
B1dot = 1./(1+exp(-alpha*(t-lambda1)));
f1 = rho1^(3/2)*exp(-g1*B1/2).*sqrt(B1dot)./(rho1+exp(-g1*B1)).^2;

f1 = (rho1^2 * exp(2*g1*B1) .* B1dot)^(1/6)./(rho1*exp(g1*B1) + 1).^(3/6);

f1 = (rho1 * exp(g1 * B1) .* sqrt(B1dot))./(rho1 * exp(g1*B1) + 1).^(3/2);
k = 0.5;

rbar_ = f1 .* ( (u(1).^H1)./(u(1).^H1 + s1^H1) );
y1 = (u(2)^2) ./ (u(2)^2 + k^2);
s = [0; rbar_; f1 .* (u(2)^1) ./ (u(2)^1 + k^1)];
% s = [0; rbar_; f1 .* y1];
end
 
function [c,f,s] = fb_pde(r,t,u,dudx,D,s0,s1,H1,J2,Z2,b2,alpha,lambda2,rho2,g2)
c = [1; 1; 1];
f = [D; 0; 0] .* dudx(1); % makes pde into ode w/ diff. source, check docs

B2 = t + log((1+exp(-alpha*(t-lambda2)))/(1+exp(alpha*lambda2)))/alpha;
phi2 = rho2./(rho2+exp(-g2*B2));
B2dot = 1./(1+exp(-alpha*(t-lambda2)));
f2 = rho2^(3/2)*exp(-g2*B2/2).*sqrt(B2dot)./(rho2+exp(-g2*B2)).^2;
k = 0.5;

% f2 = (rho2^2 * exp(2*g2*B2) .* B2dot)^(1/2)./(rho2*exp(g2*B2) + 1).^(3/2);

f2 = (rho2 * exp(g2 * B2) .* sqrt(B2dot))./(rho2 * exp(g2*B2) + 1).^(3/2);

rbar_ = f2 .* ( (u(1).^H1)./(u(1).^H1 + s1.^H1)  + b2 * ( (u(2).^J2)./(u(2).^J2 + Z2.^J2) ) ) ;
y2 = (u(2)^2) ./ (u(2)^2 + k^2);
s = [0; rbar_; f2 .* (u(2)^1) ./ (u(2)^1 + k^1)];
end


function [c,f,s] = fbqs_pde(r,t,u,dudx,D,s0,H3,J4,s3,a3,b4,Z4,alpha,lambda4,rho4,g4)
c = [1; 1; 1];
f = D .* [dudx(1); dudx(2); 0];

B4 = t + log((1+exp(-alpha*(t-lambda4)))/(1+exp(alpha*lambda4)))/alpha;
phi4 = rho4./(rho4+exp(-g4*B4));
B4dot = 1./(1+exp(-alpha*(t-lambda4)));
f4 = rho4^(3/2)*exp(-g4*B4/2).*sqrt(B4dot)./(rho4+exp(-g4*B4)).^2;
k = 0.5;

% f4 = (rho4^2 * exp(2*g4*B4) .* B4dot)^(1/6)./(rho4*exp(g4*B4) + 1).^(3/6);

f4 = (rho4 * exp(g4 * B4) .* sqrt(B4dot))./(rho4 * exp(g4*B4) + 1).^(3/2);

y4 = (u(2)^2) ./ (u(2)^2 + k^2);

s = [0; f4 .* ( a3*(u(1).^H3)./(u(1).^H3 + s3.^H3) + b4*((u(2).^J4)./(u(2).^J4 + Z4^J4)) ); ...
     f4 .* (u(2)^1) ./ (u(2)^1 + k^1)];
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


