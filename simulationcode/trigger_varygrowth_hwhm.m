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


% R = 30; % mm
% r = [0 logspace(-2,log10(R),1e3)]';%logspace(-3,3,1e3)';
% M = length(r);
% 
% T = 10; % h
% t = [0 logspace(-3,log10(T),1e2)]';%logspace(-3,2,1e2);
% N = length(t);

R = 30; % mm
% r = [0 logspace(-2,log10(R),1e3)]';%logspace(-3,3,1e3)';
% r = logspace(-2,log10(R),1e4)';%logspace(-3,3,1e3)';
r = linspace(0,15,1500)';
M = length(r);

T = 10; % h, determined by experiments 
% t = [0 logspace(-3,log10(T),1e2)]';%logspace(-3,2,1e2);
% t = logspace(-3,log10(T),1e2)';%logspace(-3,2,1e2);
t = linspace(0.01,10,1000)'; 
N = length(t);


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



m = 1;
opts = [];

betalst = [1/6,1/3,1/2,1];

thresh = 0.001;

[~,ind] = min(abs(t-3.5));

color = {'#BBE6B3','#7EC77D','#3D8E4F'};

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

index = 1; % denotes which case to run, i = [control, PF only, IS only, ISPF]


beta = betalst(1);



for ii = 1:3

    
    
    if index == 1

        % control
        Tcontrol_ = readtable('fitted neggrowth_new.xlsx', ...
            'PreserveVariableNames',true,'Range','B:D');
        Tcontrol = Tcontrol_(:,ii);
        Tcontrol = table2array(Tcontrol);
        g1 = Tcontrol(1);
        lambda1 = Tcontrol(3);
        rho1 = Tcontrol(3)/Tcontrol(4);

        % control
        solcontrol = pdepe(m,@(r,t,u,dudx)control_pde(r,t,u,dudx,D,s0,H1,s1,alpha,lambda1,rho1,g1,beta),...
            @step_ic,@step_bc,r,t,opts);
        vcontrol = solcontrol(:,:,2)';

        vcontroly = solcontrol(:,:,3)';

        % half-max distance - control
        vcontrolnorm = vcontroly./(ones(length(r),1)*vcontroly(1,:));
        [~,i] = min(abs(vcontrolnorm - .5),[],1);
        chistarcontrol = r(i);
        loglog(t,chistarcontrol,color=color{ii})

        hold on

    elseif index == 2

        % positive feedback

        Tpf_ = readtable('fitted pfgrowth_new.xlsx', ...
            'PreserveVariableNames',true,'Range','B:D');
        Tpf = Tpf_(:,ii);
        Tpf = table2array(Tpf);
        g2 = Tpf(1);
        lambda2 = Tpf(5);
        rho2 = 1/Tpf(6);

        % positive feedback
        solfb = pdepe(m,@(r,t,u,dudx)fb_pde(r,t,u,dudx,D,s0,s1,H1,J2,Z2,b2,alpha,lambda2,rho2,g2,beta),...
            @step_ic,@step_bc,r,t,opts);
        vfb = solfb(:,:,2)';

        vfby = solfb(:,:,3)';

        % half-max distance - positive feedback
        vfbnorm = vfby./(ones(length(r),1)*vfby(1,:));
        [~,i] = min(abs(vfbnorm - .5),[],1);
        chistarfb = r(i);

        loglog(t,chistarfb,color=color{ii})
        hold on



    elseif index == 3

        % quorum sensing

        Tqs_ = readtable('fitted qsgrowth_new.xlsx', ...
            'PreserveVariableNames',true,'Range','B:D');
        Tqs = Tqs_(:,ii);
        Tqs = table2array(Tqs);
        g3 = Tqs(1);
        lambda3 = Tqs(5);
        rho3 = exp(Tqs(3))/exp(Tqs(4));

        % quorum sensing
        solqs = pdepe(m,@(r,t,u,dudx)qs_pde(r,t,u,dudx,D,s0,H3,s3,a3,alpha,lambda3,rho3,g3,beta),...
            @step_ic,@step_bc,r,t,opts);
        vqs = solqs(:,:,2)';

        vqsy = solqs(:,:,3)';

        % half-max distance - quorum sensing
        vnormqs = vqsy./(ones(length(r),1)*vqsy(1,:));
        [~,i] = min(abs(vnormqs - .5),[],1);
        chistarqs = r(i);


        loglog(t,chistarqs,color=color{ii})
        hold on

    elseif index == 4

        % positive feedback + quorum sensing (trigger wave)
        Tpfqs_ = readtable('fitted twgrowth_new.xlsx', ...
            'PreserveVariableNames',true,'Range','B:D');
        Tpfqs = Tpfqs_(:,ii);
        Tpfqs = table2array(Tpfqs);
        g4 = Tpfqs(1);
        lambda4 = Tpfqs(5);
        rho4 = exp(Tpfqs(3))/exp(Tpfqs(4));

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
titlestr = "\beta = " + beta;
if index == 1; title('IS- PF-, ' + titlestr); end
if index == 2; title('IS- PF+, ' + titlestr); end
if index == 3; title('IS+ PF-, ' + titlestr); end
if index == 4; title('IS+ PF+, ' + titlestr); end
legval = [0.2,0.6,1.8];
legendStrings = "OD $ = $ " + string(legval);
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

dir = '../fig/simulation/';

if index == 1; matfilename=sprintf('varygrowth_control_hwhm%.2g',beta); end 
if index == 2; matfilename=sprintf('varygrowth_pf_hwhm%.2g',beta); end 
if index == 3; matfilename=sprintf('varygrowth_is_hwhm%.2g',beta); end 
if index == 4; matfilename=sprintf('varygrowth_ispf_hwhm%.2g',beta); end 

txt = regexprep(matfilename,'\.',''); 
saveas(f1,[dir txt],'pdf')

toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [c,f,s] = qs_pde(r,t,u,dudx,D,s0,H3,s3,a3,alpha,lambda3,rho3,g3,beta)
c = [1; 1; 1];
f = D .* [dudx(1); dudx(2); 0];

B3 = t + log((1+exp(-alpha*(t-lambda3)))./(1+exp(alpha*lambda3)))/alpha;

B3dot = 1./(1+exp(-alpha*(t-lambda3)));

n3 = ( (rho3 * exp(g3*B3) .* B3dot)./(rho3 * exp(g3*B3) + 1).^2 )^beta ...
    .* ( ( rho3 * exp(g3*B3) ) ./( rho3*exp(g3*B3) + 1 ) )^(1-beta);

z3 = (u(1).^H3) ./ (u(1).^H3 + s3.^H3);
s = [0; n3 .* a3 * z3; n3 .* u(2)];

end

function [c,f,s] = control_pde(r,t,u,dudx,D,s0,H1,s1,alpha,lambda1,rho1,g1,beta)
c = [1; 1; 1];
f = [D; 0; 0] .* dudx(1); %  makes pde into ode w/ diff. source, check docs

B1 = t + log((1+exp(-alpha*(t-lambda1)))/(1+exp(alpha*lambda1)))/alpha;

B1dot = 1./(1+exp(-alpha*(t-lambda1)));

n1 = ( (rho1 * exp(g1*B1) .* B1dot)./(rho1 * exp(g1*B1) + 1).^2 )^beta ...
    .* ( ( rho1 * exp(g1*B1) ) ./( rho1*exp(g1*B1) + 1 ) )^(1-beta);

rbar_ = n1 .* ( (u(1).^H1)./(u(1).^H1 + s1^H1) );
s = [0; rbar_; n1 .* u(2)];
end

function [c,f,s] = fb_pde(r,t,u,dudx,D,s0,s1,H1,J2,Z2,b2,alpha,lambda2,rho2,g2,beta)
c = [1; 1; 1];
f = [D; 0; 0] .* dudx(1); % makes pde into ode w/ diff. source, check docs

B2 = t + log((1+exp(-alpha*(t-lambda2)))/(1+exp(alpha*lambda2)))/alpha;

B2dot = 1./(1+exp(-alpha*(t-lambda2)));


n2 = ( (rho2 * exp(g2*B2) .* B2dot)./(rho2 * exp(g2*B2) + 1).^2 )^beta ...
    .* ( ( rho2 * exp(g2*B2) ) ./( rho2*exp(g2*B2) + 1 ) )^(1-beta);

rbar_ = n2 .* ( (u(1).^H1)./(u(1).^H1 + s1.^H1)  + b2 * ( (u(2).^J2)./(u(2).^J2 + Z2.^J2) ) ) - 0.05*u(2) ;
s = [0; rbar_; n2 .* u(2)];
end

function [c,f,s] = fbqs_pde(r,t,u,dudx,D,s0,H3,J4,s3,a3,b4,Z4,alpha,lambda4,rho4,g4,beta)
c = [1; 1; 1];
f = D .* [dudx(1); dudx(2); 0];

B4 = t + log((1+exp(-alpha*(t-lambda4)))/(1+exp(alpha*lambda4)))/alpha;

B4dot = 1./(1+exp(-alpha*(t-lambda4)));


n4 = ( (rho4 * exp(g4*B4) .* B4dot)./(rho4 * exp(g4*B4) + 1).^2 )^beta ...
    .* ( ( rho4 * exp(g4*B4) ) ./( rho4*exp(g4*B4) + 1 ) )^(1-beta);


s = [0; n4 .* ( a3*(u(1).^H3)./(u(1).^H3 + s3.^H3) + b4*((u(2).^J4)./(u(2).^J4 + Z4^J4)) ); ...
     n4 .* u(2)];
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
