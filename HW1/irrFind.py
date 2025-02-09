import numpy as np
from scipy.optimize import fsolve

def irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod):
    cash = np.array(cashFlowVec)
    length = cash.size

    def npv(r):
        total_npv = 0
        for i in range(length):
            total_npv += cash[i] / (1 + r / (12 / compoundPeriod)) ** (i * cashFlowPeriod / compoundPeriod)
        return total_npv

    irr_result = fsolve(npv, 0)[0]
    return irr_result



