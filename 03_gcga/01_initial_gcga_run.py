import ase, ase.io
from ase.optimize import BFGSLineSearch, BFGS
from ase.db import connect
from time import sleep

import numpy as np
import random
import torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

import os

# --- 1. FAIRChem MLIP 계산기 설정 ---
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
from gocia.ga.popGrandCanon import PopulationGrandCanonical
from gocia.utils.ase import geomopt_iterate
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
my_calc = FAIRChemCalculator(predictor, task_name="omat")

# --- 2. Substrate Relax (이미 최적화된 구조이므로 Single-point로 에너지 바로 획득) ---
base = ase.io.read("sub-POSCAR")
base.calc = my_calc  # 계산기 장착

# GA 파라미터
totConf = 3000
minConf = 1000
popSize = 50 # 부모가 50개이므로 popSize도 50으로 확장 권장
subsPot = base.get_potential_energy()  # Relax 생략하고 현 상태의 에너지 바로 획득

E_Ag = -2.084104 # cluster의 chemical potential 투입

chemPotDict = {
     'Ag': E_Ag,
}

# 1. 슬랩의 최상단 높이 확인
z_max_sub = base.positions[:, 2].max()
# 2. 생성 영역 설정 (슬랩 상단 + 1.5A 에서 시작하여 15A 두께의 공간 확보)
z_min_gen = 8
z_max_gen = 25 # 필요에 따라 15.0을 조절하세요
zLim = [z_min_gen, z_max_gen]
# 3. 셀 크기(Lattice parameters) 추출
cell = base.get_cell()
x_max = cell[0, 0] # 첫 번째 격자 벡터의 X 길이
y_max = cell[1, 1] # 두 번째 격자 벡터의 Y 길이 (Cartesian 기준)
xyzLims = np.array([
    [0, x_max],    # X 범위
    [0, y_max],    # Y 범위
    [z_min_gen, z_max_gen]  # Z 범위 (자동 계산된 영역)
])
print(f'GOCIA 가동 중... PID: {os.getpid()}')
print(f'자동 계산된 Z Limits: {zLim}')
print(f'자동 계산된 셀 크기: X={x_max:.2f}, Y={y_max:.2f}')


# --- 3. 데이터베이스로부터 GA 인구 초기화 ---
# 주의: 00_prep_workflow.py 를 통해 gcga.db에 부모 세대가 먼저 입력되어 있어야 작동합니다.
if not os.path.exists('gcga.db'):
    print("❌ 에러: 'gcga.db' 파일이 디렉토리에 없습니다. 00_prep_workflow.py를 먼저 가동하세요!")
    exit()

pop = PopulationGrandCanonical(
    gadb='gcga.db',
    substrate='sub-POSCAR',
    popSize = popSize,
    convergeCrit=popSize*10,
    subsPot = subsPot,
    chemPotDict = chemPotDict,
    zLim = zLim,
)

# --- 4. GA 세대 탐색 재개(Restart) 로직 ---
try:
    list_file = os.listdir('.')
    list_job = [f for f in list_file if f[0] == 's' and '.' not in f and f != 'scratch']
    if 'sub-POSCAR' in list_job: list_job.remove('sub-POSCAR')
    list_jid = [int(j[1:]) for j in list_job]
    kidnum = max(list_jid)
    print(f'Restarting the search from kidnum = {kidnum}')
except:
    kidnum = 0
    print('New search! Starting from kidnum = 0')

# 최초 1회 Population 초기화 (우리가 수동 입력한 50개를 정식 GA 인구로 선별)
if kidnum == 0:
    print('Initializing the population with custom parents in gcga.db...')
    pop.initializeDB()
    pop.natural_selection()

# --- 5. 메인 진화 루프 (GCGA Evolution Loop) ---
cwd = os.getcwd()

pop.reinitializeDB() 

while not pop.is_converged() or kidnum < minConf:
    if 'STOP' in os.listdir():
        print('STOP REQUESTED BY USER')
        exit()
    if kidnum > totConf:
        print('MAX # SAMPLE REACHED')
        exit()

    # 5-1. 다음 번호 폴더 생성 및 이동
    kidnum += 1
    kiddir = f's{str(kidnum).zfill(6)}'
    try:
        os.mkdir(kiddir)
    except FileExistsError:
        pass
    os.chdir(kiddir)

    # 5-2. Offspring 생성 (Crossover & Mutation)
    kid = None
    print(f'\n--- Generating Child {kiddir} ---')
    while kid is None:
        kid = pop.gen_offspring_box(
            mutRate=0.5,
            xyzLims=xyzLims,
            transVec=[[-1,1,-2,2,-3,3,-4,4,-6,6,-12,12],[-1,1,-2,2,-3,3,-6,6]]
        )
    
    # 초기 이완(Hooke) 및 POSCAR 저장
    kid.preopt_hooke(cutoff=1.2, toler=0.1)
    kid.write('POSCAR')

    # 5-3. MLIP 구조 최적화 (Relaxation)
    print(f'[{kiddir}] MLIP Relaxation running...')
    kid_opt = geomopt_iterate(
        kid.get_allAtoms(), 
        my_calc,
        optimizer='BFGSLineSearch',
        fmax=1e-1, 
        relax_steps=500, 
        substrate='../sub-POSCAR'
    )
    
    if kid_opt is None:
        print(f"⚠️ {kiddir} 최적화 실패 (Force exceeded threshold 등). 이 자손을 건너뜁니다.")
        os.chdir(cwd)
        continue
        
    # Force 연산 결과 후처리
    f64 = np.asarray(kid_opt.get_forces(), dtype=np.float64)
    if kid_opt.calc is not None:
        kid_opt.calc.results["forces"] = f64

    # 5-4. 업데이트 및 생존자 경쟁 (Natural Selection)
    os.chdir(cwd) # 원래의 작업 디렉토리로 복귀
    
    pop.add_aseResult(kid_opt, workdir=kiddir)
    pop.natural_selection()
    
    print(f'-> {kiddir} 처리 완료 및 Population 갱신.')
    sleep(1)

if pop.is_converged():
    print('\n🎉 CONVERGED! GLOBAL MINIMUM FOUND!')
else:
    print('\n🏁 TERMINATED!')
