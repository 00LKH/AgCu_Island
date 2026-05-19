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

# ==============================================================================
# ⚠️ [계산기 장착 영역] 사용자분의 계산기를 여기에 장착해 주세요!
# ==============================================================================
# 예시:
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

input_dir = "Ag_32"
relaxed_dir = "Ag_32_relaxed"
os.makedirs(relaxed_dir, exist_ok=True)

vasp_files = sorted(glob.glob(os.path.join(input_dir, "*.vasp")))
energies = []

for i, file_path in enumerate(vasp_files):
    print(f"[{i+1}/{len(vasp_files)}] {os.path.basename(file_path)} 최적화 중...")
    
    atoms = read(file_path)
    atoms.calc = calc
    
    # 구조 완화 실행 (fmax는 필요에 따라 조절)
    opt = BFGSLineSearch(atoms, logfile=None)
    opt.run(fmax=1e-2, steps=5000)
    
    # 에너지 및 구조 저장
    e = atoms.get_potential_energy()
    energies.append(e)
    
    write(os.path.join(relaxed_dir, f"relaxed_{os.path.basename(file_path)}"), atoms, format="vasp")
    print(f"   -> 완화 완료! 에너지: {e:.4f} eV")

# 화학 포텐셜 계산 (Ag32 클러스터이므로 원자 수 32)
num_atoms = 32
mu_list = [E / num_atoms for E in energies]

print("\n" + "="*50)
print(f"평균 Ag 1개당 화학 포텐셜 (Average): {np.mean(mu_list):.6f} eV/atom")
print(f"최소 Ag 1개당 화학 포텐셜 (Minimum): {np.min(mu_list):.6f} eV/atom")
print("="*50)
