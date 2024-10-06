import numpy as np
def calculate_saturation_factor(qs , ps):
    sats  = np.array([0, 0], dtype = float)
    top = (qs[0]*ps[1] -qs[1]*ps[0])
    for  i in range(2):
        sats[i] = top / ((ps[1] - ps[0])*qs[(i + 1)%2])
    qm = qs[0] * qs[1] * (ps[0] - ps[1] ) / (ps[0] * qs[1] - ps[1] * qs[0])
    return qs ,ps , sats ,qm