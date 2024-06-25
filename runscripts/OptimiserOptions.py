"""
==============================================================================
Optimiser Options definition
==============================================================================
@File    :   OptimiserOptions.py
@Date    :   2023/04/02
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================


def getOptOptions(alg, outputDir, majIterLimit, hessianUpdate, initPenalty, stepLimit, feasibility, optimality):
    if alg.lower() == "snopt":
        return {
            "Print frequency": 1000,
            "Summary frequency": 10000000,
            "Major feasibility tolerance": feasibility,
            "Major optimality tolerance": optimality,
            "Verify level": 0,  # -1,
            "Major iterations limit": majIterLimit,
            "Minor iterations limit": 150000000,
            "Iterations limit": 100000000,
            "Major step limit": stepLimit,
            "Nonderivative linesearch": None,
            "Linesearch tolerance": 0.9,
            "Difference interval": 1e-6,
            "Function precision": 1e-10,
            "New superbasics limit": 2000,
            "Penalty parameter": initPenalty,
            "Scale option": 1,
            "Hessian updates": hessianUpdate,
            "Print file": os.path.join(outputDir, "SNOPT_print.out"),
            "Summary file": os.path.join(outputDir, "SNOPT_summary.out"),
        }

    if alg.lower() == "ipopt":
        return {
            "max_iter": majIterLimit,
            "constr_viol_tol": feasibility,
            "nlp_scaling_method": "gradient-based",
            # "mu_init": 1e-1,
            "acceptable_tol": optimality,
            "acceptable_iter": 0,
            "tol": optimality,
            # "nlp_scaling_method": "none",
            "print_level": 0,
            "output_file": os.path.join(outputDir, "IPOPT.out"),
            "file_print_level": 5,
            "mu_strategy": "adaptive",
            "limited_memory_max_history": hessianUpdate,
            "corrector_type": "primal-dual",
            "print_user_options": "yes",
        }

    if alg.lower() == "nlpqlp":
        return {
            # NLPQL Options
            "accuracy": optimality,  # Convergence Accurancy
            "accuracyQP": 1e-14,  # Convergence Accurancy for QP
            "stepMin": 1e-3,  # Minimum step length
            # "maxFun": 20,  # Maximum Number of Function Calls During Line Search
            "maxIt": majIterLimit,  # Maximum Number of Iterations
            "maxNM": 5,  # Maximum stack size for non-monotone line search
            "rho": 1.0,  # Factor scaling identify for IFAIL=2
            "iPrint": 2,  # Output Level (0 - None, 1 - Final, 2 - Major, 3 - Major/Minor, 4 - Full)
            "mode": 0,  # NLPQL Mode (0 - Normal Execution, 1 to 18 - See Manual)
            "iOut": 6,  # Output Unit Number
            "lMerit": True,  # Merit Function Type (True - L2 Augmented Penalty, False - L1 Penalty)
            "lQl": False,  # QP Subproblem Solver (True - Quasi-Newton, False - Cholesky)
            "iFile": os.path.join(outputDir, "NLPQLP.out"),  # Output File Name
        }

    if "paropt" in alg.lower() and "mma" not in alg.lower():
        if "sl1" in alg.lower():
            tr_accept_step_strategy = "penalty_method"
        elif "filter" in alg.lower():
            tr_accept_step_strategy = "filter_method"

        return {
            "algorithm": "tr",
            "output_level": 0,
            "norm_type": "l2",
            "tr_init_size": 1e-1,
            "tr_min_size": 1e-1,
            "tr_max_size": 100.0,
            "tr_eta": 0.25,
            "tr_infeas_tol": feasibility,
            "tr_l1_tol": optimality,
            "tr_linfty_tol": 0.0,
            "tr_adaptive_gamma_update": True,
            "tr_accept_step_strategy": tr_accept_step_strategy,
            "tr_use_soc": True,
            "output_file": os.path.join(outputDir, "ParOpt.out"),
            "tr_output_file": os.path.join(outputDir, "ParOpt.tr"),
            "penalty_gamma": 50.0,
            "qn_subspace_size": hessianUpdate,
            "qn_type": "bfgs",
            "qn_diag_type": "inner_yts_over_sts",
            "abs_res_tol": 1e-12,
            "starting_point_strategy": "affine_step",
            "barrier_strategy": "mehrotra_predictor_corrector",
            "tr_steering_barrier_strategy": "mehrotra_predictor_corrector",
            "tr_steering_starting_point_strategy": "affine_step",
            "use_line_search": True,
            "use_quasi_newton_update": True,
            "tr_max_iterations": majIterLimit,
            "abs_step_tol": 1e-10,
            "qn_update_type": "damped_update",
            "function_precision": 1e-8,
        }
    if "paropt" in alg.lower() and "mma" in alg.lower():
        return {
            "algorithm": "mma",
            "output_level": 0,
            "norm_type": "l2",
            "max_major_iters": 100,
            "mma_infeas_tol": feasibility,
            "mma_l1_tol": optimality,
            "mma_linfty_tol": 0.0,
            "output_file": os.path.join(outputDir, "ParOpt.out"),
            "mma_output_file": os.path.join(outputDir, "ParOpt.mma"),
            # "penalty_gamma": 50.0,
            "qn_subspace_size": hessianUpdate,
            "qn_type": "bfgs",
            "qn_diag_type": "inner_yts_over_sts",
            "abs_res_tol": 1e-12,
            "starting_point_strategy": "affine_step",
            "barrier_strategy": "mehrotra_predictor_corrector",
            "use_line_search": True,
            "use_quasi_newton_update": True,
            "mma_max_iterations": majIterLimit,
            "abs_step_tol": 1e-10,
            "qn_update_type": "damped_update",
            "function_precision": 1e-8,
            "mma_use_constraint_linearization": True,
        }
