
import matplotlib

matplotlib.use("Agg")

import numpy as np
import random
import torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

from ase import Atoms

from agox import AGOX
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from ase.optimize import BFGSLineSearch
from agox.generators import RandomGenerator

# Manually set seed and database-index
seed = 42
database_index = 0

##############################################################################
# Calculator
##############################################################################
import os
# from fairchem.core import pretrained_mlip, FAIRChemCalculator
# os.environ['HF_TOKEN'] = '<set in .env at project root>'
# MODEL_NAME  = 'uma-s-1p1'
# # MODEL_NAME  = 'uma-m-1p1'

# predictor = pretrained_mlip.get_predict_unit(model_name = MODEL_NAME, device="cuda")
# calc = FAIRChemCalculator(predictor, task_name="omat")

from ase.calculators.emt import EMT
calc = EMT()

##############################################################################
# System & general settings:
##############################################################################

from ase.io import write
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_ag", type=int, default=12, help="Number of Ag atoms")
args = parser.parse_args()

n_ag = args.n_ag

print(f"==================================================")
print(f"Starting calculation for Ag{n_ag}")
print(f"==================================================")

# 원자 밀도 유지 (16개일 때 부피 6^3 = 216 -> 원자당 13.5 A^3)
vol = 20 * n_ag
L = vol**(1/3)

template_L = L * 2
template = Atoms("", cell=np.eye(3) * template_L)
confinement_cell = np.eye(3) * L
confinement_corner = np.array([L/2, L/2, L/2])

environment = Environment(
    template=template,
    symbols=f"Ag{n_ag}",
    confinement_cell=confinement_cell,
    confinement_corner=confinement_corner,
)

# Database
db_path = f"db_Ag{n_ag}.db"
if os.path.exists(db_path):
    os.remove(db_path) # 기존 DB가 있다면 삭제
    
database = Database(filename=db_path, order=3)

##############################################################################
# Search Settings:
##############################################################################

random_generator = RandomGenerator(**environment.get_confinement(), environment=environment, order=1)

# Wont relax fully with steps:5 - more realistic setting would be 100+.
evaluator = LocalOptimizationEvaluator(
    calc,
    gets={"get_key": "candidates"},
    optimizer = BFGSLineSearch,
    optimizer_run_kwargs={"fmax": 1e-2, "steps": 500},
    store_trajectory=False,
    order=2,
    constraints=environment.get_constraints(),
)

##############################################################################
# Let get the show running!
##############################################################################

agox = AGOX(random_generator, database, evaluator, seed=seed)

agox.run(N_iterations=100)

db = Database(filename=db_path)
db.restore_to_memory()
candidates = db.get_all_candidates()

traj_name = f"all_candidates_Ag{n_ag}.traj"
write(traj_name, candidates)
print(f"{traj_name} 파일 생성이 완료되었습니다.\n")
