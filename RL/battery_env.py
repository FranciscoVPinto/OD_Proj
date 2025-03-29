import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class BatteryEnv(gym.Env):
    def __init__(self, consumers_file, producers_file, agg_interval=480, eta_c=0.95, eta_d=0.95,
                 E_max=2500, C_grid=0.1, penalty_grid=2.0, penalty_waste=0.2):
        super(BatteryEnv, self).__init__()

        # Parâmetros
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.E_max = E_max
        self.initial_soc = E_max * 0.5
        self.C_grid = C_grid
        self.penalty_grid = penalty_grid
        self.penalty_waste = penalty_waste
        self.agg_interval = agg_interval
        self.timestep_ratio = agg_interval // 15

        # Dados
        consumers = pd.read_excel(consumers_file, skiprows=1)
        producers = pd.read_excel(producers_file, skiprows=1)

        nrows = (len(consumers) // self.timestep_ratio) * self.timestep_ratio
        self.P_load = consumers.iloc[:nrows].groupby(consumers.index[:nrows] // self.timestep_ratio).sum().sum(axis=1).values
        self.P_prod = producers.iloc[:nrows].groupby(producers.index[:nrows] // self.timestep_ratio).sum().sum(axis=1).values
        self.T = len(self.P_load)

        # Espaço de observação
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]),
                                            high=np.array([E_max, np.inf, np.inf]),
                                            dtype=np.float32)

        # Espaço de ação
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.SoC = self.initial_soc
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.SoC, self.P_load[self.t], self.P_prod[self.t]], dtype=np.float32)

    def step(self, action):
        frac_c, frac_d = action
        P_load = self.P_load[self.t]
        P_prod = self.P_prod[self.t]

        P_c_max = min(P_prod, (self.E_max - self.SoC) / self.eta_c)
        P_c = frac_c * P_c_max

        P_d_max = self.SoC
        P_d = frac_d * P_d_max
        P_d = min(P_d, self.SoC)

        self.SoC += self.eta_c * P_c - P_d / self.eta_d
        self.SoC = np.clip(self.SoC, 0, self.E_max)

        P_batt = self.eta_d * P_d
        P_grid = max(0, P_load - P_batt)
        P_waste = max(0, P_prod - P_c)

        # Recompensa: penaliza uso da rede e desperdício solar
        reward = - (P_grid * (self.C_grid + self.penalty_grid) + P_waste * self.penalty_waste)

        self.t += 1
        done = self.t >= self.T - 1
        return self._get_obs(), reward, done, False, {}

    def render(self):
        pass  # opcional
