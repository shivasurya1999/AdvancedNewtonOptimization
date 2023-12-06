clc; clear; close all;
 
epsilon = 1e-4; %Newton min keeps running until the diff. between step updates is less than this (TBD: include function value update epsilon in this)
max_iter = 2000; %max iterations of Newton min. Function exits even if not convergent beyong this 
max_iter_ls = 20; %max number of itrations of line search  
max_iter_tr = 50; %max number of itrations of trust region 
alpha_eigen = 1e-4; %amount added to min_eigenvalue to make Hessian +ve def 
alpha_ls = 1e-4; %parameter used in rhs of Armijo condition in line search and Trust region approach 


%Define the Extended Rosenbrock Function for n = 2
syms x1 x2;
f_rosenbrock = 10*(x2 - x1^2)^2 + (1 - x1)^2;

% Gradient of the Extended Rosenbrock Function
grad_rosenbrock = gradient(f_rosenbrock, [x1, x2]);

% Hessian of the Extended Rosenbrock Function
H_rosenbrock = hessian(f_rosenbrock, [x1, x2]);

xf_rosenbrock = [x1,x2];

% Initial guess for the optimization
x0_rosenbrock = [-1.2; 1];
% x0_rosenbrock = [0.5; 0.8];
% x0_rosenbrock = [5; -5];
% x0_rosenbrock = [100; 100];

x_vals = [x0_rosenbrock];

errors = [];

% Run Rosenbrock
implementMultiNewtonMin(f_rosenbrock, H_rosenbrock, grad_rosenbrock, xf_rosenbrock, x0_rosenbrock, epsilon, x_vals, errors, max_iter, max_iter_ls, max_iter_tr, alpha_eigen, alpha_ls); 



% % Define the Extended Powell Singular Function for n = 4
% syms x1 x2 x3 x4;
% f_powell = (x1 + 10*x2)^2 + 5*(x3 - x4)^2 + (x2 - 2*x3)^4 + 10*(x1 - x4)^4;
% 
% % Gradient of the Extended Powell Singular Function
% grad_powell = gradient(f_powell, [x1, x2, x3, x4]);
% 
% % Hessian of the Extended Powell Singular Function
% H_powell = hessian(f_powell, [x1, x2, x3, x4]);
% 
% xf_powell = [x1,x2,x3,x4];
% 
% % Initial guess for the optimization
% % x0_powell = [3; -1; 0; 1];
% % x0_powell = [0.5; 0.6; -0.5; -0.6];
% % x0_powell = [7; 8; -8; -10];
% x0_powell = [50; -30; -10; 90];
% 
% x_vals = [x0_powell];  
% 
% errors = [];  % Create a list to store the error
% 
% % Run Powell
% implementMultiNewtonMin(f_powell, H_powell, grad_powell, xf_powell, x0_powell, epsilon, x_vals, errors, max_iter); 



% %Define the Wood Function for n = 4
% syms x1 x2 x3 x4;
% f_wood = 100*(x1^2 - x2)^2 + (x1 - 1)^2 + (x3 - 1)^2 + 90*(x3^2 - x4)^2 + ...
%           10.1*((x2 - 1)^2 + (x4 - 1)^2) + 19.8*(x2 - 1)*(x4 - 1);
% 
% %Gradient of the Wood Function
% grad_wood = gradient(f_wood, [x1, x2, x3, x4]);
% 
% %Hessian of the Wood Function
% H_wood = hessian(f_wood, [x1, x2, x3, x4]);
% 
% xf_wood = [x1,x2,x3,x4];
% 
% %Initial guess for the optimization
% % x0_wood = [-3; -1; -3; -1];
% % x0_wood = [0.5; 1.2; -1.3; -0.4];
% % x0_wood = [10; -10; -6; 15];
% x0_wood = [50; -30; 25; -80];
% 
% x_vals = [x0_wood];  
% 
% errors = [];  % Create a list to store the errors

% % Run Wood
% implementMultiNewtonMin(f_wood, H_wood, grad_wood, xf_wood, x0_wood, epsilon, x_vals, errors, max_iter); 


function [is_pos_def, eigenvalues] = isSafelyPosDef(H_current)
    % Check if H_current is symmetric
    if H_current ~= H_current'
        is_pos_def = false;
        eigenvalues = [];
        return;
    end

    % Get the eigenvalues of H_current
    eigenvalues = eig(H_current);

    % Check if all eigenvalues are positive
    is_pos_def = all(isAlways(eigenvalues > 0));

end


function lambda_k = quadraticFitting(f_xk, f_dot_xk, f_xkplus)
    lambda_k = -(f_dot_xk)/(2*(f_xkplus-f_xk-f_dot_xk));
end

function lambda_k_new = cubicFitting(f_xk, f_dot_xk, lambda_prev, lambda_two_prev, f_lprev, f_ltwoprev)
    % Coefficients
    c = f_dot_xk;
    d = f_xk;
    
    % Calculation of a and b using matrices
    const = 1 / (lambda_prev - lambda_two_prev);
    mat1 = [(1/lambda_prev^2) (-1/lambda_two_prev^2); (-lambda_two_prev/lambda_prev^2) (lambda_prev/lambda_two_prev^2)];
    mat2 = [f_lprev - f_xk - c*lambda_prev; f_ltwoprev - f_xk - c*lambda_two_prev];

    mat = const * mat1 * mat2;
    a = mat(1);
    b = mat(2);

    % Checking for real roots
    discriminant = b^2 - 3*a*c;
    if discriminant < 0
        disp('No real roots for cubic equation. Setting lambda_k_new to 0.');
        lambda_k_new = 0;
    else
        lambda_k_new = (-b + sqrt(discriminant)) / (3 * a);
    end
    
end


function [x_new, iter] = BacktrackingSearch(f, x, x_current, H_x_current, grad_x_current, alpha, max_iter_ls)
    lambda_k = 1; 
    p_k = - H_x_current \ grad_x_current; %Newton direction 

    iter = 1;

    %initialize lambda_prev and lambda_two_prev randomly (changes in loop, so can initialize to anything)
    lambda_prev = lambda_k;
    lambda_two_prev = lambda_k;

    f_xkplus_values = []; 

    while true

        lhs_x = x_current + lambda_k*p_k;
        subsStructLhs = struct;
        subsStructRhs = struct;
    
        for i = 1:length(x)
            subsStructLhs.(char(x(i))) = lhs_x(i);
            subsStructRhs.(char(x(i))) = x_current(i);
        end
    
        f_xk = vpa(subs(f,subsStructRhs));
        f_xkplus = vpa(subs(f, subsStructLhs));
        f_dot_xk = grad_x_current'*p_k; 
        lhs_comp = f_xkplus;
        rhs_comp = f_xk + alpha*lambda_k*grad_x_current'*p_k;

        % Append the computed f_xkplus value to the array
        f_xkplus_values = [f_xkplus_values, f_xkplus];


        if lhs_comp < rhs_comp || iter < max_iter_ls   %Armijo condition
            % fprintf('Debug: Currently at iteration %d\n', iter);
            break
        else
            % debugMessageNew = 'This is a newww debug message.';
            % disp(debugMessageNew);

            if iter < 3 %we do quadratic fitting until we have lamda_prev and lambda_two_prev
                if iter >= 2
                    lambda_two_prev = lambda_prev;
                end
                
                lambda_prev = lambda_k;
                lambda_k = quadraticFitting(f_xk,f_dot_xk,f_xkplus);
                
            else
                f_lprev = f_xkplus_values(end);
                f_ltwoprev = f_xkplus_values(end-1);
                lambda_k_new = cubicFitting(f_xk,f_dot_xk,lambda_prev,lambda_two_prev,f_lprev,f_ltwoprev);
                lambda_two_prev = lambda_prev;
                lambda_prev = lambda_k;
                lambda_k = lambda_k_new;
            end

        end

        iter = iter + 1;

    end

    if lambda_k/lambda_prev < 0.1 
        lambda_k = 0.1*lambda_prev;
    end

    if lambda_k/lambda_prev > 0.5
        lambda_k = 0.5*lambda_prev;
    end

    if iter < max_iter_ls
        x_new = x_current + lambda_k*p_k;
    else
        x_new = x_current;
    end

end


function [x_new, iter_tr] = TrustRegionDoglegUpdate(f, x, x_current, H_x_current, grad_x_current, alpha_ls, max_iter_tr)
    s_N = - H_x_current \ grad_x_current; %Newton direction

    grad_norm = norm(grad_x_current);
    lambda_star = (grad_norm^2)/(grad_x_current'*H_x_current*grad_x_current);

    delta_current = lambda_star*grad_norm; %we initialize tr radius as length of cauchy step  

    gamma_num = grad_norm^4;
    H_x_inv_grad_x = H_x_current \ grad_x_current; % More efficient than using inv()
    gamma_denom = (grad_x_current' * H_x_current * grad_x_current) * (grad_x_current' * H_x_inv_grad_x);
    gamma = gamma_num/gamma_denom;

    eta = 0.8*gamma + 0.2;
    s_Nhat = eta*s_N; %scaled dogleg direction 

    iter_tr = 1;

    while iter_tr < max_iter_tr
        if delta_current < lambda_star*grad_norm
            s_CP = (-delta_current*grad_x_current)/grad_norm;
        else    
            s_CP = -lambda_star*grad_x_current;
        end


        % we need to solve for lambda in (norm(s_CP + lambda*(s_Nhat-s_CP)))^2 = delta_current^2
        % Expand the norm and square it
        lambda = 0;
        a = norm(s_Nhat - s_CP)^2;
        b = 2 * (s_Nhat - s_CP)' * s_CP;
        c = norm(s_CP)^2 - delta_current^2;

        % Solve the quadratic equation
        discriminant = b^2 - 4*a*c;
        if discriminant >= 0
            lambda_1 = (-b + sqrt(discriminant)) / (2*a);
            lambda_2 = (-b - sqrt(discriminant)) / (2*a);

            % Choose the appropriate lambda within [0,1]
            lambda_options = [lambda_1, lambda_2];
            for i = 1:length(lambda_options)
                if lambda_options(i) >= 0 && lambda_options(i) <= 1
                    lambda = lambda_options(i);
                    break;
                end
            end
        end

        x_new = x_current + s_CP + lambda*(s_Nhat - s_CP);

        %code to check for alpha condition 
        lhs_x = x_new;
        subsStructLhs = struct;
        subsStructRhs = struct;
    
        for i = 1:length(x)
            subsStructLhs.(char(x(i))) = lhs_x(i);
            subsStructRhs.(char(x(i))) = x_current(i);
        end
    
        f_xc = vpa(subs(f,subsStructRhs));
        f_xplus = vpa(subs(f, subsStructLhs)); 
        lhs_comp = f_xplus;
        rhs_comp = f_xc + alpha_ls*grad_x_current'*(x_new - x_current); 

        if lhs_comp > rhs_comp %alpha condition fails 

            lambda_plus = quadraticFitting(f_xc, grad_x_current'*(x_new - x_current), f_xplus); %quadratic backtrack strategy (same as in ls method)
            delta_new = lambda_plus*norm(x_new - x_current);

            %we need to constrain the new delta_current between
            %0.1*delta_current and 0.5*delta_current
            if delta_new > delta_current/2
                delta_new = delta_current/2;
            end

            if delta_new < delta_current/10
                delta_new = delta_current/10;
            end

            delta_current = delta_new;

        end

        iter_tr = iter_tr + 1;

        
    end

    if iter_tr >= max_iter_tr
        x_new = x_current;
    end

end



% Function for single Newton's step
function x_new = singleStepMultiNewtonMin(f, Hf, grad_f, x, x_current, max_iter_ls, max_iter_tr, alpha_eigen, alpha_ls)
    subsStruct = struct;

    for i = 1:length(x)
        subsStruct.(char(x(i))) = x_current(i);
    end 

    % H_x_current = vpa(subs(Hf, subsStruct));
    H_x_current = simplify(vpa(subs(Hf, subsStruct)));
    grad_x_current = vpa(subs(grad_f,subsStruct));

    [is_pos_def,eigenvalues] = isSafelyPosDef(H_x_current);
    min_eigenvalue = min(eigenvalues);
    mu = min_eigenvalue + alpha_eigen;

    if ~is_pos_def 
        H_x_current = H_x_current + mu * eye(size(H_x_current));
    end
 
    % x_new = x_current - H_x_current \ grad_x_current;


    [x_new, iter_ls] = BacktrackingSearch(f, x, x_current, H_x_current, grad_x_current, alpha_ls, max_iter_ls);

    if iter_ls >= max_iter_ls
        [x_new, iter_tr] = TrustRegionDoglegUpdate(f, x, x_current, H_x_current, grad_x_current, alpha_ls, max_iter_tr);
    end

    if iter_tr >= max_iter_tr
        disp('Maximum Number of trust region iterations have been reached. Exiting');
    end
    
end


% Function to implement Newton's method
function implementMultiNewtonMin(f, Hf, grad_f, xf, x0, epsilon, x_vals, errors, max_iter, max_iter_ls, max_iter_tr, alpha_eigen, alpha_ls)
    x_current = x0;
    itr = 0;
    err = Inf;

    while itr < max_iter && err > epsilon
        x_new = singleStepMultiNewtonMin(f, Hf, grad_f, xf, x_current, max_iter_ls, max_iter_tr, alpha_eigen, alpha_ls);
        err = norm(x_new - x_current);
        x_current = x_new;
        x_vals = [x_vals, x_current];
        errors(end+1) = err;  % Add the error to the list
        itr = itr + 1;
    end
    
    % Plotting x-values
    subplot(2,1,1);
    norms = arrayfun(@(idx) norm(x_vals(:, idx)), 1:size(x_vals, 2));
    plot(norms);
    xlabel('Iterations');
    ylabel('Norm of x-values');
    title('Convergence of Wood');
    
    % Plotting the logarithm of errors
    subplot(2,1,2);
    plot(log(errors));  % Use log() function to plot logarithm of errors
    xlabel('Iterations');
    ylabel('log(error)');
    title('Log of Errors in Wood');
    
    % Print or return the result
    fprintf('The root is approximately:\n');
    disp(x_current);
end
