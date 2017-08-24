%% % ------ Solving QP using Interior Point Methods ------ %
% Apurva Badithela
% 6/14/17
% This script will input the parameters into the qcqp_solver.m
% minimize (1/2)x'*P*x + q'*x + r
% subject to: qi'*x + ri <= 0 and A*x = b

clear all
close all

% Test MPC problem for two inputs, varying time horizon length and varying
% state size. Number of outputs = no. of states
N_min = 5;
N_max = 30;
m_min = 3;
m_max = 10;

TEST = 10;
solve_time = zeros(N_max - N_min + 1, m_max - m_min + 1);
tests_passed = zeros(N_max - N_min + 1, m_max - m_min + 1);
% [N_x,N_y] - To keep track of solve times corresponding to given N and m

N_x = 1;
for N_iter = N_min:N_max    
    N_y = 1;
    for m_iter = m_min : m_max
        test_times = zeros(1,TEST);
        for test = 1:TEST
            % Test Case - Formulating a Model Predictive Control Problem as a QP:
            %     mass = 1; % kg. Mass of hovercraft.
            %     A_state = [zeros(2,2), eye(2); zeros(2,2), zeros(2,2)];
            %     B_state = [zeros(2,2); 1.0/mass*eye(2)];
            %     C_state = [1, 1, 1, 1];
            %     D_state = 0;
            %     sys = ss(A_state, B_state, C_state, D_state);
            %     vx = inst*0.5 - 50;
            %     vy = inst*0.5;
            %     z0 = [2;2;vx;vy];
            %     zf = [0; 0;0;0];
            
            k = m_iter; % Dimension of system state
            l = 2; % Dimension of control action
            N = N_iter; % Time Horizon
            n = N*(k+l); % Length of optimization variable
            
            % Generating a random continuous time stable system:
            sys = rss(m_iter, m_iter, l);
            z0 = randperm(10,k)';
            zf = zeros(k,1);
            
%             Q = 10*ones(k);
%             R = 0.01*ones(l);
%             Qf = 10*ones(k);
%             euler = 1;

            
            Q = 10*eye(k);
            R = 0.01*eye(l);
            Qf = 10*eye(k);
            euler = 1;
            
            Qtilde = euler*Q;
            Rtilde = euler*R;
            Qftilde = Qf;
            
            % ZOH discretization:
            T = 0.01; % Sample time is 0.01 s
            zoh_sys = c2d(sys, T, 'zoh');
            Atilde = zoh_sys.A;
            Btilde = zoh_sys.B;
            
            row = N*k;
            col = N*(k+l);
            t = N*(k+l) + k; % See pg. 552 BV
            
            A_dyn = zeros(row,t);
            b_dyn = zeros(row,1);
            
            set = [-Atilde, -Btilde, eye(k,k)];
            len = length(set);
            for i = 1:N
                A_dyn(1+(i-1)*k:i*k,1+(i-1)*(k+l):len+(i-1)*(k+l)) = set;
            end
            
            A_dyn = A_dyn(:,k+1:end);
            b_dyn(1:k) = Atilde*z0;
            
            % Building P0, q0, r0:
            % For MPC, q0, r0 are zero matrices
            id = zeros(l+k);
            id(1:l,1:l) = Rtilde;
            id(l+1:l+k,l+1:l+k) = Qtilde;
            P0 = zeros(t);
            
            for i = 1:N
                P0(1 + (i-1)*(l+k) : i*(l+k), 1 + (i-1)*(l+k) : i*(l+k)) = id;
            end
            P0(t-k+1:t, t-k+1:t) = Qftilde;
            
            P0 = 2*P0; % So that the objective is 1/2*x'*P0*x
            q0 = zeros(t,1);
            r0 = 0;
            
            % Building A and b matrix from dynamics and initial and final conds:
            A = zeros(row+2*k, t);
            A(1:k, 1:k) = eye(k);
            A(row+k+1:row+2*k, t-k+1:t) = eye(k);
            A(k+1: row+k, k+1:t) = A_dyn;
            b = [z0; b_dyn; zf];
            
            % Building qi's and ri's:
            u_min = -2500;
            u_max = 2500;
            
            % Inequality constraints on input:
            up = zeros(l*N,t);
            ii = 1;
            for i = 1:N
                up(ii, 1+k+(i-1)*(l+k)) = 1;
                up(ii + 1, i*(l+k)) = 1;
                ii = ii+2;
            end
            qi = [up; -up];
            ri = [-u_max*ones(N*l,1); u_min*ones(N*l,1)];
            qi = qi';
            
            QP = struct('f0', {P0, q0, r0}, 'Fi',{qi, ri, []}, 'A', {A, [], []}, 'b', {b, [], []});
            
            [x_sol, opt_sol, time, pass] = qp_solver(QP);
            if(pass)
                test_times(test) = time;
                
                % Extract control outputs and states:
                x = zeros(k, N+1);
                u = zeros(l, N);
                for i = 1:N
                    u(:,i) = x_sol(1+k+(i-1)*(l+k): i*(l+k));
                    x(:,i) = x_sol(1+(i-1)*(l+k):k+(i-1)*(l+k));
                end
                x(:, N+1) = x_sol(N*(l+k)+1: t);
                
                % Checking optimal value of objective:
                optval = 0;
                for i = 1:N
                    optval = optval + x(:,i)'*Qtilde*x(:,i) + u(:,i)'*Rtilde*u(:,i);
                end
                optval = optval + x(:,N+1)'*Qftilde*x(:,N+1);
            end

        end % End of running 10 test cases        
        tests_passed(N_x, N_y) = nnz(test_times);
        solve_time(N_x, N_y) = sum(test_times)/tests_passed(N_x, N_y); %AVg. solve time
        N_y = N_y+1;
    end % End of m_iter
    N_x = N_x + 1;
end % End of N_iter

% End for problem instances
% figure(2)
% title('Hovercraft trajectory');
% xlabel('X');
% ylabel('Y');
% grid on
% hold on
% dest = plot(0,0, 'r*');
% start = plot(z0(1), z0(2), 'b^');
% plot(z0(1), z0(2), 'k-');
% 
% for i = 1:N+1
%     pause(0.25)
%     start = plot(x(1,i), x(2,i), 'b^');
% end

figure(100);
[X,Y] = meshgrid(N_min:N_max, m_min:m_max);
Z = solve_time;
surf(X,Y,Z');
%  title('Average MPC solve times prior to pre-ordering');
xlabel('Time Horizon');
ylabel('State Size');
zlabel('Solve Time(s)');
yticks(m_min:m_max);