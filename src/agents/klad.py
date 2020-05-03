import numpy as np
import os
from src.pretraining.MultiDimensionalExpertDataset import MultiDimensionalExpertDataset


dataset = MultiDimensionalExpertDataset(expert_path=['./experts/1_StackedEnv_NU_CREHPAJBJCTU_0.0002.npz','./experts/1_StackedEnv_NU_CREHPAJBJCTU_0.005.npz'], traj_limitation=-1, batch_size=32)