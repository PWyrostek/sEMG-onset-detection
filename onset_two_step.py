from onset_aglr import onset_AGLRstep_two_step
from onset_sign_changes import onset_sign_changes


def onset_two_step_alg(data, W_1, h_2, W_2, M_2):
    FIRST_STAGE_FAULT_TOLERANCE = 100
    START = FIRST_STAGE_FAULT_TOLERANCE
    END = 5000

    left, right = onset_sign_changes(data, W_1)  # 120

    left = left or START
    right = right or END

    result = onset_AGLRstep_two_step(data, h_2, W_2, M_2, left - FIRST_STAGE_FAULT_TOLERANCE, right)  # 20,15,20
    return result
