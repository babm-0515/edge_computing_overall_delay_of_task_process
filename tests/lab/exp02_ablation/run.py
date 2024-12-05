import numpy as np

from src.tools.load_config import load_config_and_initialize_class
from src.scenario.sim import Sim
import tests.lab.panel.ctrl_sim as control_sim

np.random.seed(1)
np.set_printoptions(linewidth=np.inf)

"""
sec1(Offload+Schedule+Cache): is_serial=True, is_scheduled=True, has_caching=True, solver='cplex'
sec2(Offload+Schedule): is_serial=True, is_scheduled=True, has_caching=False, solver='cplex'
sec3(Offload+Cache): is_serial=True, is_scheduled=False, has_caching=True, solver='cplex'
sec4(Offload): is_serial=True, is_scheduled=False, has_caching=False, solver='cplex'
"""

if __name__ == '__main__':
    config_path = '../../config/exp02_ablation/exp02.ini'
    max_iot, min_iot, step = 30, 10, 5
    for num_iot in range(min_iot, max_iot + step, step):
        setting_name = 'iot_' + f'{int(num_iot)}' + '_edge_5'
        # print(setting_name)
        sim: Sim = load_config_and_initialize_class(config_path, setting_name, Sim)

        control_sim.run(sim, is_serial=True, is_scheduled=True, has_caching=False, solver='cplex')
