import os
import glob
import numpy as np
import random
import torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
from ase.io import read, write
from ase.optimize import BFGS, BFGSLineSearch
from ase.constraints import FixAtoms

# Load HF_TOKEN from .env at project root (walks up from this file)
_d = os.path.dirname(os.path.abspath(__file__))
while _d != '/':
    _p = os.path.join(_d, '.env')
    if os.path.isfile(_p):
        for _line in open(_p):
            if '=' in _line and not _line.lstrip().startswith('#'):
                _k, _v = _line.strip().split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
        break
    _d = os.path.dirname(_d)
if not os.environ.get('HF_TOKEN'):
    raise SystemExit('HF_TOKEN not set. Add HF_TOKEN=... to .env at project root.')

from fairchem.core import pretrained_mlip, FAIRChemCalculator
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omat")

input_dir = "generated_surfaces"
relaxed_dir = "generated_surfaces_relaxed"
os.makedirs(relaxed_dir, exist_ok=True)

vasp_files = sorted(glob.glob(os.path.join(input_dir, "*.vasp")))
energies = []

for i, file_path in enumerate(vasp_files):
    print(f"[{i+1}/{len(vasp_files)}] {os.path.basename(file_path)} 최적화 중...")
    
    atoms = read(file_path)
    atoms.calc = calc
    
    # 3. 구조에 Constraint(고정 제약조건) 적용
    fixed_indices = [atom.index for atom in atoms if atom.position[2] < 11.0]
    constraint = FixAtoms(indices=fixed_indices)
    atoms.set_constraint(constraint)
    
    # 확인할 수 있도록 고정된 원자 개수 출력
    print(f"   -> Total atoms: {len(atoms)}")
    print(f"   -> Fixed atoms: {len(fixed_indices)}")
    
    # 구조 완화 실행 (fmax는 필요에 따라 조절)
    opt = BFGSLineSearch(atoms, logfile='-')
    opt.run(fmax=1e-1, steps=1000)
    
    # 에너지 및 구조 저장
    e = atoms.get_potential_energy()
    energies.append(e)
    
    write(os.path.join(relaxed_dir, f"relaxed_{os.path.basename(file_path)}"), atoms, format="vasp")
    print(f"   -> 완화 완료! 에너지: {e:.4f} eV")
