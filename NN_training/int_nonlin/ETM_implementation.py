import os
from systems_and_LMI.LMI.int_3l.ETM_sat import LMI_3l_int_ETM_sat
import numpy as np

W1_name = os.path.abspath(__file__ + "/../new_weights/mlp_extractor.policy_net.0.weight.csv")
W2_name = os.path.abspath(__file__ + "/../new_weights/mlp_extractor.policy_net.2.weight.csv")
W3_name = os.path.abspath(__file__ + "/../new_weights/mlp_extractor.policy_net.4.weight.csv")
W4_name = os.path.abspath(__file__ + "/../new_weights/action_net.weight.csv")

b1_name = os.path.abspath(__file__ + "/../new_weights/mlp_extractor.policy_net.0.bias.csv")
b2_name = os.path.abspath(__file__ + "/../new_weights/mlp_extractor.policy_net.2.bias.csv")
b3_name = os.path.abspath(__file__ + "/../new_weights/mlp_extractor.policy_net.4.bias.csv")
b4_name = os.path.abspath(__file__ + "/../new_weights/action_net.bias.csv")

W1 = np.loadtxt(W1_name, delimiter=',')
W2 = np.loadtxt(W2_name, delimiter=',')
W3 = np.loadtxt(W3_name, delimiter=',')
W4 = np.loadtxt(W4_name, delimiter=',')
W4 = W4.reshape((1, len(W4)))

W = [W1, W2, W3, W4]

b1 = np.loadtxt(b1_name, delimiter=',')
b2 = np.loadtxt(b2_name, delimiter=',')
b3 = np.loadtxt(b3_name, delimiter=',')
b4 = np.loadtxt(b4_name, delimiter=',')

b = [b1, b2, b3, b4]

# lmi = LMI_3l_int_ETM(W, b)
lmi = LMI_3l_int_ETM_sat(W, b)
# alpha = lmi.search_alpha(0.2, 0, 1e-5, verbose=True)
alpha = 1
P, T, Z = lmi.solve(alpha, verbose=True)
# lmi.save_results('ETM')