# Advanced Newton optimization implementation details 

## Initializations

- **Parameter Tuning**: The code initializes tunable parameters to vary the performance of the implementation. These include:
  - Least difference between step updates.
  - Maximum number of iterations and sub-iterations for line search and trust region methods.
  - Addition to the minimum eigenvalue for Hessian positive definiteness.
  - Step size in the Armijo condition.

## Function Definition and Initialization

- **Analytical Foundations**: The function is defined using symbolic variables. Gradient and Hessian are computed using MATLAB functions.
- **Optimization Variables and Arrays**: Initial guess is set. Arrays for variable values, function evaluations, and convergence metrics are initialized.
- **Optimization Invocation**: The `implementMultiNewtonMin` function is called, provided with initialization and function parameters.

## Main Function

- **Core Logic**: `implementMultiNewtonMin` handles the multi-variable Newton optimization process.
- **Iteration and Convergence**: Iterations begin with an initial guess, assessing convergence through the norm of the step difference.
- **Subsidiary Function Calls**: `singleStepMultiNewtonMin` is used for single optimization steps, including Newton, line search, and trust region steps.
- **Termination and Visualization**: Upon meeting convergence criteria or iteration limits, the function plots convergence and performance metrics.
- **Summary**: A summary table of step types and iteration counts is displayed, along with the approximate solution.

## Single Step Minimization Function

- **Function Role**: `singleStepMultiNewtonMin` computes a single iteration within the optimization framework.
- **Initialization and Computation**: Initializes flags and counters, computes function value, gradient, and Hessian.
- **Decision Making**: Chooses the optimal step type based on the current state.
- **Hessian Adjustments**: Adjusts the Hessian for positive definiteness.
- **Line Search and Trust Region Updates**: Implements backtracking line search with a maximum sub-iteration limit, and escalates to trust region updates if necessary.

## Line Search and Polynomial Fitting Functions

- **Quadratic and Cubic Fitting**: Functions `quadraticFitting` and `cubicFitting` estimate appropriate step sizes.
- **Backtracking Line Search**: `BacktrackingSearch` uses the Armijo condition for λ adjustments and step size refinement.

## Trust Region Update with Dogleg Method

- **Initialization**: Marks entry into the trust region phase.
- **Trust Region Radius and Path Strategy**: Calculates initial radius and computes the dogleg path.
- **Iteration and Alpha Condition**: Checks the alpha condition, performs quadratic fitting for radius refinement.
- **Termination and Output**: Continues until an acceptable step is found or the maximum sub-iteration count is reached.

# Results and Report 
