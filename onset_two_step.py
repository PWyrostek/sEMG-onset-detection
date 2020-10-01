import numpy as np
from utilities import estimate_theta_0, make_plot_sign_changes
from onset_aglr import onset_AGLRstep_two_step
from onset_sign_changes import onset_sign_changes


def onset_two_step_alg(data, W_1, k_1, d_1, h_2, W_2, M_2):
    first_stage_fault_tolerance = 100
    left, right = onset_sign_changes(data, W_1, k_1, d_1)  # 200,1,0.01
    result = onset_AGLRstep_two_step(data, h_2, W_2, M_2, left - first_stage_fault_tolerance, right)  # 20,15,20
    return result
