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

from ase.io import read
from ase.db import connect

# --- 출력 위치를 ./process/ 로 통일 ---
# 입력(slab_source, vasp_dir)은 절대경로이므로 chdir 영향 없음.
# 출력(sub-POSCAR, gcga.db)만 process/ 안에 생성됨.
PROCESS_DIR = 'process'
os.makedirs(PROCESS_DIR, exist_ok=True)
os.chdir(PROCESS_DIR)
print(f'출력 디렉토리: {os.getcwd()}')

# ==========================================
# 1. Substrate (Slab) Preparation
# ==========================================
slab_source = '/home/kyunghun/05_AGOX/13/00_slab/4_Ag_slab_supercell_opt.vasp'
slab_dest = 'sub-POSCAR'

print("--- STEP 1: Preparing Substrate ---")
if os.path.exists(slab_source):
    try:
        slab_atoms = read(slab_source)
        # Slab에 해당하는 원자만 확실하게 분리 (예: 1280개)
        if len(slab_atoms) > 1280:
            slab_atoms = slab_atoms[:1280]
        slab_atoms.write(slab_dest)
        print(f"✅ {slab_dest} 파일 구축 완료.")
    except Exception as e:
        print(f"Slab 변환 에러: {e}")
        exit()
else:
    print(f"에러: {slab_source} 파일을 찾을 수 없습니다.")
    exit()

print("")

# ==========================================
# 2. VASP 파일들로부터 gcga.db 구축
# ==========================================
vasp_dir = '/home/kyunghun/05_AGOX/13/02_slab_cluster/generated_surfaces_relaxed'
db_name = 'gcga.db'

print("--- STEP 2: Loading VASP files and preparing Calculator ---")
vasp_files = sorted(glob.glob(os.path.join(vasp_dir, "*.vasp")))
if not vasp_files:
    print(f"에러: {vasp_dir} 경로에 .vasp 파일이 존재하지 않습니다.")
    exit()

print(f"총 {len(vasp_files)}개의 .vasp 파일을 발견했습니다.")

# MLIP 계산기 세팅 (에너지 계산용)
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
print("✅ MLIP 계산기 로드 완료.")

# 기존 gcga.db 삭제 후 클린 리셋
if os.path.exists(db_name):
    print(f"기존에 존재하는 {db_name}를 삭제합니다.")
    os.remove(db_name)

print(f"구조를 {db_name}에 등록합니다...")

success_count = 0
with connect(db_name) as db:
    for i, file_path in enumerate(vasp_files):
        try:
            atoms = read(file_path)
            
            # 계산기가 있는 경우 에너지 계산, 없는 경우 임시 에너지 0.0 부여
            if calc is not None:
                atoms.calc = calc
                pot_energy = atoms.get_potential_energy()
            else:
                pot_energy = 0.0
                
            # GOCIA DB에 등록
            db.write(
                atoms,
                eV = pot_energy,
                done = 1,
                alive = 1,
                name = f'parent_{os.path.basename(file_path).split(".")[0]}',
                mag = 0
            )
            success_count += 1
            print(f"[{success_count}/{len(vasp_files)}] {os.path.basename(file_path)} 등록 완료 (Energy: {pot_energy:.4f} eV)")
        except Exception as e:
            print(f"⚠️ {os.path.basename(file_path)} 처리 중 에러 발생: {e}")

if success_count > 0:
    print(f"\n🎉 {db_name} 파일 빌드 성공! ({success_count}개 데이터)")
    print("이제 'python 01_gcga_run.py' 명령어로 즉시 테스트를 진행해보실 수 있습니다!")
else:
    print("\n❌ 에러: 등록된 데이터가 전혀 없습니다.")
