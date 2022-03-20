import math

import numpy as np
import math

class Ote28Globals:


    def __init__(self):
        Ote28Globals.fwhm_sigma = 2 * math.sqrt(2.0 * math.log(2.0))
        return

    @staticmethod
    def Gauss(x, amp, fwhm, xpk):
        sigma = fwhm / Ote28Globals.fwhm_sigma
        k = (x - xpk) / sigma
        y = amp * np.exp((-k ** 2) / 2.0)
        return y

