from criteria import least_confidence
import numpy as np
pred_prob = np.array([[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.2, 0.1, 0.1, 0.5]])

least_confidence(pred_prob, 1)
