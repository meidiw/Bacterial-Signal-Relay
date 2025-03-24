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
r = [0 logspace(-1,log10(R),1e3)]';%logspace(-3,3,1e3)';
M = length(r);

T = 10; % h 
t = [0 logspace(-3,log10(T),1e3)]';%logspace(-3,2,1e2);
N = length(t);


m = 1;
opts = [];

b2list = [0.1, 1, 10];
b4list = [0.1, 1, 10];

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


% 1 -> positive feedback (b2), 2 -> trigger wave (b4)

test = 2;

if test == 1
    for ii = 1:3

        k = b2list(ii);


        % positive feedback

        solfb = pdepe(m,@(r,t,u,dudx)fb_pde(r,t,u,dudx,D,s0,s1,H1,J2,Z2,k,alpha,lambda2,rho2,g2),...
            @step_ic,@step_bc,r,t,opts);
        vfb = solfb(:,:,2)';

        vfby = solfb(:,:,3)';



        % half-max distance - positive feedback
        vfbnorm = vfby./(ones(length(r),1)*vfby(1,:));
        [~,i] = min(abs(vfbnorm - .5),[],1);
        chistarfb = r(i);
        loglog(t,chistarfb,color=color{ii})
        tval = 1;
        hold on


    end

elseif test == 2
    for ii = 1:3

        k = b4list(ii);

        % positive feedback + quorum sensing (trigger wave)

        solfbqs = pdepe(m,@(r,t,u,dudx)fbqs_pde(r,t,u,dudx,D,s0,H3,J4,s3,a3,k,Z4,alpha,lambda4,rho4,g4),...
            @step_ic,@step_bc,r,t,opts);
        vfbqs = solfbqs(:,:,2)';

        vfbqsy = solfbqs(:,:,3)';

        % half-max distance - trigger wave
        vnormfbqs = vfbqsy./(ones(length(r),1)*vfbqsy(1,:));
        [~,i] = min(abs(vnormfbqs - .5),[],1);
        chistarfbqs = r(i);
        loglog(t,chistarfbqs,color=color{ii})
        tval = 2;
        hold on


    end

else
    0;
end



% Wavefronts data
t_ = [3 10]; % h
whalf = 3 * (t_ / 3).^.5;
wone = 3 * (t_ / 3);

ydiffplot = loglog(t_,whalf,':','Color','#808080',LineWidth=2);
hold on
ybalisticplot = loglog(t_,wone,'k:',LineWidth=2);
xlim([3 10])
ylim([2 20])
yticks([2 4 6 8 10]);
yticklabels({'2', '4', '6','8', '10'})
xlabel('Time [hr]','FontSize',20)
ylabel('HWHM [mm]','FontSize',20)
if tval == 1; title('IS- PF+'); legendStrings = "$b_2 = $ " + string(b2list); end
if tval == 2; title('IS+ PF+'); legendStrings = "$b_4 = $ " + string(b4list); end
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

function [c,f,s] = qs_pde(r,t,u,dudx,D,s0,H3,s3,a3,alpha,lambda3,rho3,g3)
E = 1.6 * D;
c = [1; 1; 1];
f = [D * dudx(1); E * dudx(2); 0];

B3 = t + log((1+exp(-alpha*(t-lambda3)))./(1+exp(alpha*lambda3)))/alpha;
phi3 = rho3./(rho3+exp(-g3*B3));
B3dot = 1./(1+exp(-alpha*(t-lambda3)));
f3 = rho3^(3/2)*exp(-g3*B3/2).*sqrt(B3dot)./(rho3+exp(-g3*B3)).^2;

z3 = (u(1).^H3) ./ (u(1).^H3 + s3.^H3);
s = [0; f3 .* a3 * z3; f3 .* u(2)];

end

function [c,f,s] = control_ode(r,t,u,dudx,D,s0,H1,s1,alpha,lambda1,rho1,g1)
E = 1.6 * D;
c = [1; 1; 1];
f = [D * dudx(1); 0; 0];

B1 = t + log((1+exp(-alpha*(t-lambda1)))/(1+exp(alpha*lambda1)))/alpha;
phi1 = rho1./(rho1+exp(-g1*B1));
B1dot = 1./(1+exp(-alpha*(t-lambda1)));
f1 = rho1^(3/2)*exp(-g1*B1/2).*sqrt(B1dot)./(rho1+exp(-g1*B1)).^2;

rbar_ = f1 .* ( (u(1).^H1)./(u(1).^H1 + s1^H1) );
s = [0; rbar_; f1 .* u(2)];
end
 
function [c,f,s] = fb_pde(r,t,u,dudx,D,s0,s1,H1,J2,Z2,b2,alpha,lambda2,rho2,g2)
E = 1.6 * D;
c = [1; 1; 1];
f = [D * dudx(1); 0; 0];

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
