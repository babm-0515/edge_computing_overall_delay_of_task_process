import numpy as np

from src.tools.load_config import load_config_and_initialize_class
from src.scenario.sim import Sim
import tests.lab.panel.ctrl_sim as control_sim

np.random.seed(3)
np.set_printoptions(linewidth=np.inf)


if __name__ == '__main__':
    config_path = '../../config/demo/test_load_env.ini'
    sim: Sim = load_config_and_initialize_class(config_path, 'Settings', Sim)

    control_sim.run(sim, is_serial=True, is_scheduled=True, linear=True, deletion=True)
