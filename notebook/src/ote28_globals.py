import math

import numpy as np
import math

class Globals:


    def __init__(self):
        Globals.fwhm_sigma = 2 * math.sqrt(2.0 * math.log(2.0))
        Globals.ref_radius = 1.9      # Encircled energy reference radius
        return

    @staticmethod
    def Gauss(x, amp, fwhm, xpk):
        sigma = fwhm / Globals.fwhm_sigma
        k = (x - xpk) / sigma
        y = amp * np.exp((-k ** 2) / 2.0)
        return y

