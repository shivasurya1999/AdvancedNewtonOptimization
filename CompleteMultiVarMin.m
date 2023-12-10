clc; clear; close all;
 
epsilon = 1e-4; %Newton min keeps running until the diff. between step updates is less than this (TBD: include function value update epsilon in this)
max_iter = 1500; %max iterations of Newton min. Function exits even if not convergent beyong this 
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
x0_rosenbrock = -5000*[-1.2; 1];
% x0_rosenbrock = [0.5; 0.8];
% x0_rosenbrock = [5; -5];
% x0_rosenbrock = [100; 100];

x_vals = [x0_rosenbrock];
f_vals = [];  % Initialize an empty array to store function values
errors = [];

% Run Rosenbrock
implementMultiNewtonMin(f_rosenbrock, H_rosenbrock, grad_rosenbrock, xf_rosenbrock, x0_rosenbrock, epsilon, x_vals, errors, max_iter, max_iter_ls, max_iter_tr, alpha_eigen, alpha_ls, f_vals); 



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
% x0_powell = [50; -30; -0.0005; 0.009];
% 
% x_vals = [x0_powell];  
% f_vals = [];  % Initialize an empty array to store function values
% errors = [];  % Create a list to store the error
% 
% % Run Powell
% implementMultiNewtonMin(f_powell, H_powell, grad_powell, xf_powell, x0_powell, epsilon, x_vals, errors, max_iter, max_iter_ls, max_iter_tr, alpha_eigen, alpha_ls,f_vals); 



% %Define the Wood Function for n = 4
% syms x1 x2 x3 x4;
% f_wood = 100*(x1^2 - x2)^2 + (x1 - 1)^2 + (x3 - 1)^2 + 90*(x3^2 - x4)^2 + ...
%           10.1*((x2 - 1)^2 + (x4 - 1)^2) + 19.8*(x2 - 1)*(x4 - 1) ;
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
% x0_wood = [-3; -1; -3; -1];
% % x0_wood = [0.5; 1.2; -1.3; -0.4];
% % x0_wood = [10; -10; -6; 15];
% % x0_wood = [500; -30; -250; 80];
% 
% x_vals = [x0_wood];  
% f_vals = [];  % Initialize an empty array to store function values
% errors = [];  % Create a list to store the errors
% 
% % Run Wood
% implementMultiNewtonMin(f_wood, H_wood, grad_wood, xf_wood, x0_wood, epsilon, x_vals, errors, max_iter, max_iter_ls, max_iter_tr, alpha_eigen, alpha_ls,f_vals); 



% %Define custom Function for n = 4
% syms x1 x2 x3 x4;
% 
% % f_custom = (1.5 - x1 * (1 - x2))^2 + (2.25 - x1 * (1 - x2^2))^2 + (2.625 - x1 * (1 - x2^3))^2 + x3^2 + x4^2; %beale function (solution : (3,0.5,0,0)
% 
% % f_custom = -cos(x1) .* cos(x2) .* exp(-((x1 - pi).^2 + (x2 - pi).^2));
% 
% % f_custom = sin(x1 * x2) + x1^2 - cos(x3 * x4) + exp(-(x2^2 + x3^2)) + (x4 - x1)^4;
% 
% % f_custom = sin(x1 * x2) + cos(x3 * x4) + x1^2 * x4 - x2 * x3^2 + ...
% %     exp(-x1 * x4) + x3 * x2^2 - 5 * x1 * x4 + 3 * x2 * x3 - x1^2 * x3^2 + ...
% %     2 * x2^2 * x4^2 - 4 * x1 + 3;
% 
% f_custom = x1^2 + (x4-1)^2 + x2^2 + x3^2;
% 
% 
% %Gradient of the custom Function
% grad_custom = gradient(f_custom, [x1, x2, x3, x4]);
% 
% %Hessian of the custom Function
% H_custom = hessian(f_custom, [x1, x2, x3, x4]);
% 
% xf_custom = [x1,x2,x3,x4];
% 
% %Initial guess for the optimization
% x0_custom = [-3; -1; -3; -1];
% % x0_custom = [0.5; 1.2; -1.3; -0.4];
% % x0_custom = [10; -10; -6; 15];
% % x0_custom = [500; -30; -250; 80];
% 
% x_vals = [x0_custom];  
% f_vals = [];  % Initialize an empty array to store function values
% errors = [];  % Create a list to store the errors
% 
% % Run custom function
% implementMultiNewtonMin(f_custom, H_custom, grad_custom, xf_custom, x0_custom, epsilon, x_vals, errors, max_iter, max_iter_ls, max_iter_tr, alpha_eigen, alpha_ls,f_vals); 




function [is_pos_def, eigenvalues] = isSafelyPosDef(H_current) %returns true if Hessian is positive definite else returns false 
    % Check if H_current is symmetric
    if ~isequal(H_current, H_current')
        is_pos_def = false;
        eigenvalues = [];
        return;
    end

    % Get the eigenvalues of H_current
    eigenvalues = eig(H_current);

    % Check if all eigenvalues are positive
    is_pos_def = all(eigenvalues > 0);
end



function lambda_k = quadraticFitting(f_xk, f_dot_xk, f_xkplus) %fits quadratic polynomial for line search or trust region lambda update 
    lambda_k = -(f_dot_xk)/(2*(f_xkplus-f_xk-f_dot_xk));
end

function lambda_k_new = cubicFitting(f_xk, f_dot_xk, lambda_prev, lambda_two_prev, f_lprev, f_ltwoprev) %fits cubic polynomial for line search lambda update
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
        % If the discriminant is negative, take the real part of the complex solution
        lambda_k_new = -b / (3*a);
        % fprintf('After cubic fitting lambda_k_new: %f\n', lambda_k_new);
    else
        % Standard solution (real)
        lambda_k_new = (-b + sqrt(discriminant)) / (3*a);
        % fprintf('After cubic good fit lambda_k_new: %f\n', lambda_k_new);
    end
    
end


function [x_new, iter] = BacktrackingSearch(f, x, x_current, H_x_current, grad_x_current, alpha, max_iter_ls)  %performs line search using Armijo condition
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

      
        fprintf('ls fxk: %f, fxkplus: %f\n', f_xk, f_xkplus);


        if lhs_comp < rhs_comp || iter >= max_iter_ls   %Armijo condition

            if iter == 1
                fprintf('Took Newton step with number of subiterations %d\n',iter);
                % disp('\n')
            else
                fprintf('Took Line Search step with number of subiterations %d\n',iter-1);
            end

            break

        else

            if iter < 3 %we do quadratic fitting until we have lamda_prev and lambda_two_prev
                if iter >= 2
                    lambda_two_prev = lambda_prev;
                end
                
                lambda_prev = lambda_k;
                lambda_k = quadraticFitting(f_xk,f_dot_xk,f_xkplus);

                % fprintf('After quad fitting lambda_k: %f, lambda_prev: %f\n', lambda_k, lambda_prev);
                
            else
                f_lprev = f_xkplus_values(end);
                f_ltwoprev = f_xkplus_values(end-1);
                lambda_k_new = cubicFitting(f_xk,f_dot_xk,lambda_prev,lambda_two_prev,f_lprev,f_ltwoprev);
                %check whether lambda_k_new is within bounds and update 
                if(lambda_k_new < 0.1*lambda_k)
                    lambda_k_new = 0.1*lambda_k;
                end
                if(lambda_k_new > 0.5*lambda_k)
                    lambda_k_new = 0.5*lambda_k;
                end
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

    x_new = x_current + lambda_k*p_k;

    % x_new = x_current;
   

end


function [x_new, iter_tr] = TrustRegionDoglegUpdate(f, x, x_current, H_x_current, grad_x_current, alpha_ls, max_iter_tr) %performs trust region update using double dogleg method
    fprintf('We have entered Trust Region so keep calm and Trust \n');
    s_N = - H_x_current \ grad_x_current; %Newton direction

    grad_norm = norm(grad_x_current);
    lambda_star = (grad_norm^2)/(grad_x_current'*H_x_current*grad_x_current);

    delta_current = abs(lambda_star*grad_norm); %we initialize tr radius as length of cauchy step  

    gamma_num = grad_norm^4;
    H_x_inv_grad_x = H_x_current \ grad_x_current; % More efficient than using inv()
    gamma_denom = (grad_x_current' * H_x_current * grad_x_current) * (grad_x_current' * H_x_inv_grad_x);
    gamma = gamma_num/gamma_denom;

    eta = 0.8*gamma + 0.2;
    % fprintf('In TR gamma: %f  eta: %f\n', gamma, eta);

    s_Nhat = eta*s_N; %scaled dogleg direction 

    iter_tr = 1;

    while iter_tr < max_iter_tr
        if delta_current < abs(lambda_star*grad_norm)
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

        % fprintf('In TR lambda1 is: %f lambda2 is: %f\n', lambda_options(1),lambda_options(2));

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

        % fprintf('In TR lhs_comp: %f, rhs_comp: %f\n', lhs_comp, rhs_comp);

        if lhs_comp > rhs_comp %alpha condition fails 

            lambda_plus = quadraticFitting(f_xc, grad_x_current'*(x_new - x_current), f_xplus); %quadratic backtrack strategy (same as in ls method)
            delta_new = abs(lambda_plus*norm(x_new - x_current));

            %we need to constrain the new delta_current between
            %0.1*delta_current and 0.5*delta_current
            if delta_new > delta_current/2
                delta_new = delta_current/2;
            end

            if delta_new < delta_current/10
                delta_new = delta_current/10;
            end

            % fprintf('In TR delta_curr: %f, delta_new: %f\n', delta_current, delta_new);

            delta_current = delta_new;

        else
            fprintf('Took Trust Region step with number of subiterations %d\n',iter_tr);
            break;

        end

        iter_tr = iter_tr + 1;

        
    end

end



% Function for single Newton's step (calls LS and TR methods if applicable)
function [x_new,exit_flag,f_x_current,iter_ls,iter_tr, stepType] = singleStepMultiNewtonMin(f, Hf, grad_f, x, x_current, max_iter_ls, max_iter_tr, alpha_eigen, alpha_ls)
    exit_flag = false; % Initialize the flag
    stepType = ''; % Initialize step type

    iter_ls = 0;
    iter_tr = 0;

    subsStruct = struct;

    for i = 1:length(x)
        subsStruct.(char(x(i))) = x_current(i);
    end 

    % H_x_current = vpa(subs(Hf, subsStruct));
    f_x_current = simplify(vpa(subs(f, subsStruct)));
    H_x_current = simplify(vpa(subs(Hf, subsStruct)));
    grad_x_current = vpa(subs(grad_f,subsStruct));

    [is_pos_def,eigenvalues] = isSafelyPosDef(H_x_current);
    min_eigenvalue = min(eigenvalues);
    mu = abs(min_eigenvalue) + alpha_eigen;

    if ~is_pos_def 
        H_x_current = H_x_current + mu * eye(size(H_x_current));
    end

    % fprintf('after conditionin the hessian is_pos_def:%f, mu: %f',is_pos_def,mu)
 
    % x_new = x_current - H_x_current \ grad_x_current;


    [x_new, iter_ls] = BacktrackingSearch(f, x, x_current, H_x_current, grad_x_current, alpha_ls, max_iter_ls);

    if iter_ls == 1 % Condition for Newton step
        stepType = 'Newton';
    end
        
    if iter_ls >= 2 % Condition for Line Search
        stepType = 'Line Search';
    end

 
    if iter_ls >= max_iter_ls

        [x_new, iter_tr] = TrustRegionDoglegUpdate(f, x, x_current, H_x_current, grad_x_current, alpha_ls, max_iter_tr);

        if iter_tr > 0 % Condition for Trust Region 
            stepType = 'Trust Region';    
        end

        if iter_tr >= max_iter_tr
            exit_flag = true;
            disp('Maximum Number of trust region iterations have been reached. Exiting');
        end

    end


end


%Main function that implements Multi-variable Newton optimization 
function implementMultiNewtonMin(f, Hf, grad_f, xf, x0, epsilon, x_vals, errors, max_iter, max_iter_ls, max_iter_tr, alpha_eigen, alpha_ls, f_vals)
    x_current = x0;
    itr = 0;
    err = Inf;
    stepSummaryInfo = {}; %stores information of the type of strategy being used for the summary table displayed at the end 
    lastStepType = '';
    lastStepCount = 0;

    while itr < max_iter && err > epsilon
        [x_new, exit_flag, f_x_current, iter_ls, iter_tr, stepType] = singleStepMultiNewtonMin(f, Hf, grad_f, xf, x_current, max_iter_ls, max_iter_tr, alpha_eigen, alpha_ls);
        err = norm(x_new - x_current);
        x_current = x_new;
        x_vals = [x_vals, x_current];
        errors(end+1) = err;  
        f_vals(end+1) = f_x_current;
        itr = itr + 1;

        % Check if the step type has changed
        if ~strcmp(lastStepType, stepType)
            if lastStepCount > 0
                % Store the previous step summary
                stepSummaryInfo{end+1} = {lastStepType, lastStepCount};
            end
            lastStepType = stepType;
            lastStepCount = 1;
        else
            lastStepCount = lastStepCount + 1;
        end

        if exit_flag
            break;
        end
    end

    % Add the last step type and count
    if lastStepCount > 0
        stepSummaryInfo{end+1} = {lastStepType, lastStepCount};
    end

    % Plotting and displaying the root
    subplot(3,1,1);
    norms = arrayfun(@(idx) norm(x_vals(:, idx)), 1:size(x_vals, 2));
    plot(norms);
    xlabel('Iterations');
    ylabel('Norm of x-values');
    title('Convergence of x-values');
    text(itr, norms(itr), sprintf('Exit point coordinates: (%d, %f)', itr, norms(itr)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');

    subplot(3,1,2);
    plot(f_vals);
    xlabel('Iterations');
    ylabel('Function Value');
    title('Function Value over Iterations');
    text(itr, f_vals(itr), sprintf('Exit point coordinates: (%d, %f)', itr, f_vals(itr)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');


    subplot(3,1,3);
    log_errors = log(errors);
    plot(log_errors);
    xlabel('Iterations');
    ylabel('log(error)');
    title('Log of Errors');
    text(itr, log_errors(itr), sprintf('Exit point coordinates: (%d, %f)', itr, log_errors(itr)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');


    fprintf('The root is approximately:\n');
    disp(x_current);

    % Print the step summary after plotting and displaying the root
    fprintf('\nSummary Table:\n');
    fprintf('SNo\tStep Taken\t\t# of Iterations\n');
    for i = 1:length(stepSummaryInfo)
        fprintf('%d\t%s\t\t\t%d\n', i, stepSummaryInfo{i}{1}, stepSummaryInfo{i}{2});
    end
end

