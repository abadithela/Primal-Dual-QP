function[x_sol, opt_sol, time, pass] = qp_solver(QP)
%% % ===== QCQP Solver ===== %
% qcqp_solver.m takes input from qcqp.m and implements the primal dual method
% to solve the QCQP. The inner loop for implementing the Newton step is
% qcqp_ctrstep.m and it is called from this function
% Parameters are set in this inner loop

%% ===== SETUP ===== %

P0 = QP(1).f0;
q0 = QP(2).f0;
r0 = QP(3).f0;

% 3D matrices with i being the third dimension 
q  = QP(1).Fi; 
r  = QP(2).Fi; 

A  = QP.A;
b  = QP.b;

%% ===== PARAMETERS AND TOLERANCES ====== %
ALPHA = 0.01;
BETA = 0.5;
FTOL = 1e-3;
MAX_ITER = 40;
EPSTOL = 1e-6;
chk = 0; % Primal feasibility checker
PARAMS = [ALPHA, BETA, FTOL, MAX_ITER, EPSTOL, chk];
[m, n] = size(A);
sz = size(q,2);
MU = 20;
t = 1;
GAP = n/t;
HIST = []; gap_hist = [];
x_sol = []; opt_sol = [];

%% ===== Finding a starting point in the domain - PHASE 1 ===== %
% x_start = A\b;
cvx_begin;
    variable s;
    variable x_start(n);
    minimize(s);
    q'*x_start+r <= s;
    A*x_start == b;
cvx_end;
%% ===== Calling Infeasible Start Newton Method ===== %
profile on
tic;
s_lamq = 0;
lam_start = ones(sz,1);
for i = 1:sz
    s_lamq = s_lamq + lam_start(i)*q(:,i);
end
nu_start = -(A')\(P0*x_start + q0 + s_lamq);

while(GAP > EPSTOL)
    [x_star, lam_star, nu_star, r2_hist] = qp_ctrstep(QP, x_start, lam_start, nu_start,t,PARAMS);
    if(isempty(x_star) || any(isnan(x_star)))
        pass = 0;
        break;
    end
    x_start = x_star;
    lam_start = lam_star;
    nu_start = nu_star;
    t = MU*t; 
    GAP = n/t;
    HIST = [HIST r2_hist]; gap_hist = [gap_hist GAP];
end
pass = 1; % Program converges
time = toc;
profile off
% figure (1)
% [xx, yy] = stairs(cumsum(HIST), gap_hist);
% semilogy(xx,yy,'bo-');
% xlabel('Newton Iterations');
% ylabel('Gap');
% title('QP Solver - Total primal dual iterations');

x_sol = x_start;
opt_sol = 0.5*x_sol'*P0*x_sol;
end