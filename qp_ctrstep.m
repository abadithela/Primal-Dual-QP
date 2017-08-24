function [xopt, lamopt, nuopt, r2_hist] = qp_ctrstep(QP, x0, lam0, nu0,t, PARAMS)
% ===== qcqp_ctrstep ====== %
% Inner loop implementing infeasible start Newton's method 
% a Quadratically constrained quadratic program (QCQP)
% Inputs: 
% QCQP: Struct defining QCQP with fields: f0, Fi, A, b
% QCQP.f0: 1-by-3 cell array of {P0, q0, r0} where 
% the objective is given by: (1/2)x'*P0*x + q0'*x + r0
% QCQP.Fi: m-by-3 cell array of form {Pi, qi, ri} (i = 1,..,m)
% This defines our quadratic constraints: (1/2)x'*Pi*x + qi'*x + ri <= 0
% QCQP.A:1-by-1 cell array containing m-by-n matrix on the left hand side of equality constraint
% b: 1-by-1 cell array containing m-by-1 matrix on the right hand side of the equality constraint
% x0: n-by-1 vector of the central point computed from the previous step
% nu0: m-by-1 vector of the dual point computed in the previous Newton step
% QCQP.t: Strength of barrier
% PARAMS: Parameters and tolerances for the Newton's method

% Parameters --
ALPHA = PARAMS(1);
BETA = PARAMS(2);


% Tolerances  --
FTOL = PARAMS(3); % Feasibility Tolerance
MAX_ITER = PARAMS(4); % Max. number of iterations
EPSTOL = PARAMS(5); 
TIMETOL = 10000;
MU = 10;

P0 = QP(1).f0;
q0 = QP(2).f0;
r0 = QP(3).f0;

% 3D matrices with i being the third dimension
q  = QP(1).Fi; % 2D matrix with ith column having qi
r  = QP(2).Fi; % Vector with ith element being ri

A  = QP.A;
b  = QP.b;

[m, n] = size(A);
sz = size(q,2); % Number of constraints
xopt = []; lamopt = []; nuopt = []; r2_hist = []; count = 0; x = x0; nu = nu0; lam = lam0;

% Primal-Dual Interior Point Algorithm (pg. 610 - 612 BV):
count = 1;
while(count < MAX_ITER)
    % Computing f and Df at point x:
    f = q'*x + r;
    Df = q';
    % Determining surrogate duality gap:
    SGAP = -f'*lam;

    t = MU*sz/SGAP;
    sum_Hpd = 0;
    % Computing Hessian and LHS in eq. 11.55:
    for i = 1 : sz
        sum_Hpd = sum_Hpd + lam(i)/(-f(i))*q(:,i)*q(:,i)';
    end
    
    H_pd = P0 + sum_Hpd;
    kkt = [H_pd, A'; A zeros(m)];
    
    % Computing RHS in eq. 11.55:
    df0 = P0*x + q0;

    r_dual = df0 + Df'*lam + A'*nu; 
    r_cent = -diag(lam)*f - (1.0/t)*ones(sz,1);
    r_pri = A*x - b;
    res = [r_dual; r_cent; r_pri];
    
    % Break out of search:
    if((norm(r_pri) <= FTOL) && (norm(r_dual) <= FTOL) && (SGAP <= EPSTOL))
        break;
    end
    
    sum_top = 0;
    for i = 1 : sz
        sum_top = sum_top + (1.0)/(-f(i))*q(:,i);
    end
    r_top = df0 + (1.0/t)*sum_top + A'*nu;
    rhs = -[r_top; r_pri];
    
    % Finding Primal-Dual Search Direction:
    % skkt = sparse(kkt);
    NT = kkt\rhs;

    % Finding fill-in reducing permutation P to facilitate LDL'
    % factorization:
%     KKT = size(kkt,1);
%     min_fill = KKT; % Starting as large as possible
%     P = eye(KKT);
%     for i = 1:KKT
%         P = sparse(1:KKT, combination, ones(1,KKT));
%         [L, D, p] = ldl(P*kkt*P');
%         fill = nnz(L) - nnz(tril(P'*kkt*P));
%         if(fill < min_fill); min_fill = fill; min_L = L; min_D = D; end
%     end
    
    dx_pd = NT(1:n);
    dnu_pd = NT(n+1:end);
    dlam_pd = -(diag(f))\diag(lam)*Df*dx_pd + diag(f)\r_cent;
    
    % Modified Backtracking Line Search:
    pos_lam = zeros(size(lam)); % lam_i/dlam_i
    for i = 1:sz
        if(dlam_pd(i) < 0)
            pos_lam(i) = -lam(i)/dlam_pd(i);
        end
    end
    pos_lam(pos_lam == 0) = [];
    
    if(isempty(pos_lam))
        smax = 1;
    else
        smax = min(1, min(pos_lam));
    end
    
    s = 0.99*smax;
    x_new = x + s*dx_pd;
    lam_new = lam + s*dlam_pd;
    nu_new = nu + s*dnu_pd;
    f_new = q'*x_new + r;
    
    exit_timer = 0;
    
    while(any(f_new >= FTOL))
        if(exit_timer > TIMETOL)
            break;
        end
        s = BETA*s;
        x_new = x + s*dx_pd;
        lam_new = lam + s*dlam_pd;
        nu_new = nu + s*dnu_pd;
        f_new = q'*x_new + r;
        exit_timer = exit_timer + 1;
    end
    
    if(exit_timer > TIMETOL) % If problem fails to converge
        x = []; lam = []; nu = []; count = [];
        break;
    end
    
    df0_new = P0*x_new + q0;
    Df_new = q';
    r_dual_new = df0_new + Df_new'*lam_new + A'*nu_new; 
    r_cent_new = -diag(lam_new)*f_new - (1.0/t)*ones(sz,1);
    r_pri_new = A*x_new - b;
    res_new = [r_dual_new; r_cent_new; r_pri_new];
    
    exit_timer = 0;
    while((norm(res_new) - (1-ALPHA*s)*norm(res)) > FTOL)
        s = BETA*s;
        x_new = x + s*dx_pd;
        lam_new = lam + s*dlam_pd;
        nu_new = nu + s*dnu_pd;
        res_new = [x_new; lam_new; nu_new];   
        if(exit_timer > TIMETOL)
            break;
        end
    end
    
    if(exit_timer > TIMETOL) % If problem fails to converge
        x = []; lam = []; nu = []; count = [];
        break;
    end
    % Update
    x = x + s*dx_pd;
    nu = nu + s*dnu_pd;
    lam = lam + s*dlam_pd;
    
    count  = count + 1;
end


xopt = x;
lamopt = lam;
nuopt = nu;
r2_hist = count;

end


% Objective and quadratic constraints
% f0  = @(x) (t*(1.0/2*x'*P0*x  + q0'*x  + r0));
% f   = cell(m,1);
% grd = zeros(n,m);
% Hess= zeros(n,n,m);

% Feasibility Requirements --
% if(chk == 1)
%     assert(norm(A*x0 - b) < FTOL) % Assert only after primal feasibility has been acheived
% end
% 
% for i = 1:m
%     Pi = cell2mat(P(i));
%     qi = cell2mat(q(i));
%     ri = cell2mat(r(i));
%     f{i} = @(x) (1.0/2*x'*Pi*x + qi'*x + ri);
%     % fi(x_star) < 0
%     assert(f{i}(x0) < FTOL); 
% end
% 
% while(count < MAX_ITER)
%     count = count + 1;
%     
%     % Hessian and Gradient
%     g_phi = zeros(n,1);
%     H_phi = zeros(n,n);
%     gf = P0*x + q0;
%     Hf = P0;
%     
%     for i = 1:m
%         Pi = cell2mat(P(i));
%         qi = cell2mat(q(i));
%         ri = cell2mat(r(i));
%         fi = cell2mat(f{i}(x));
%         grd(:,i) = Pi*x + qi;
%         Hess(:,:,i) = Pi; 
%         g_phi = g_phi + grd(:,i);
%         H_phi = H_phi + Hess(:,:,i);
%     end
%     
%     g = gf + g_phi;
%     H = Hf + H_phi;
%     
%     % Solve KKT system
%     KKT = [H, A'; A, zeros(m,m)];
%     
%     if(chk == 1)
%         r = [g + A'*nu; zeros(m,1)];
%     else
%         r = [g + A'*nu; A*x - b];
%     end
%     
%     sol = -KKT\r;
%     xnt = sol(1:n); % = xpd
%     nunt = sol(n+1:end); % = nupd  
%     
%     % Residual
%     r2 = norm(r,2);
%     r2_hist = [r2_hist r2];
%     
%     % Exit from Newton's Method
%     if(norm(A*x - b) < FTOL)
%         chk = 1;
%     end    
%     if(chk == 1 && r2 <= EPSTOL)
%         break;
%     end
%     
%     % Line Search
%     l = 1;
%     
%     [r2_dual, r2_pri] = residual(QCQP, x + l*xnt, nu + l*nunt);     
%     r2_nxt = norm([r2_dual, r2_pri],2);
%     
%     while(r2_nxt - r2 + alpha*t*r2 > FTOL)
%         [r2_dual, r2_pri] = residual(QCQP, x + l*xnt, nu + l*nunt);     
%         r2_nxt = norm([r2_dual, r2_pri],2);
%         l = l*beta;
%     end
%     
%     % Update:
%     x = x + l*xnt;
%     nu = nu + l*nunt;
% end
% 
% 
% end