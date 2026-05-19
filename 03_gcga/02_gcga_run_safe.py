"""
01_gcga_run.py 의 VRAM 누수 안전 버전.

- 모든 입출력 파일(gcga.db, sub-POSCAR, s000xxx/)을 ./process/ 하위에 둠.
  Python files / batch scripts 는 03_gcga/ 에 유지.
- 기존 process/s000xxx 폴더를 스캔하여 자동 재개
- 매 iteration 종료 시 gc.collect() + torch.cuda.empty_cache() 로 VRAM 정리
- CUDA OOM 발생 시 해당 자손을 건너뛰고 메모리 해제 후 다음 자손으로 진행
- 치명적 예외 시 os.execv 로 자기 자신을 재실행 → DB/폴더 상태로부터 자동 재개

사용법: python -u 02_gcga_run_safe.py   (03_gcga/ 에서 실행)
사전 준비:
    mkdir -p process
    mv gcga.db sub-POSCAR gmid s000* process/
"""

import ase, ase.io
from ase.optimize import BFGSLineSearch, BFGS
from ase.db import connect
from time import sleep

import gc
import sys
import traceback
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

# --- 0. 작업 디렉토리를 ./process/ 로 전환 ---
# gcga.db / sub-POSCAR / s000xxx / gmid 등 런타임 데이터는 모두 여기에 위치.
# 이후 모든 상대경로(gcga.db, sub-POSCAR, ../sub-POSCAR 등)는 process/ 기준으로 해석됨.
PROCESS_DIR = 'process'
os.makedirs(PROCESS_DIR, exist_ok=True)
os.chdir(PROCESS_DIR)
print(f'작업 디렉토리: {os.getcwd()}')

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
base.calc = my_calc

totConf = 3000
minConf = 1000
popSize = 50
subsPot = base.get_potential_energy()

E_Ag = -2.084104
chemPotDict = {
    'Ag': E_Ag,
}

z_max_sub = base.positions[:, 2].max()
z_min_gen = 8
z_max_gen = 25
zLim = [z_min_gen, z_max_gen]
cell = base.get_cell()
x_max = cell[0, 0]
y_max = cell[1, 1]
xyzLims = np.array([
    [0, x_max],
    [0, y_max],
    [z_min_gen, z_max_gen]
])
print(f'GOCIA 가동 중... PID: {os.getpid()}')
print(f'자동 계산된 Z Limits: {zLim}')
print(f'자동 계산된 셀 크기: X={x_max:.2f}, Y={y_max:.2f}')


# --- 3. DB로부터 GA population 초기화 ---
if not os.path.exists('gcga.db'):
    print("❌ 에러: 'gcga.db' 파일이 디렉토리에 없습니다. 00_prep_workflow.py를 먼저 가동하세요!")
    exit()

pop = PopulationGrandCanonical(
    gadb='gcga.db',
    substrate='sub-POSCAR',
    popSize=popSize,
    convergeCrit=popSize*10,
    subsPot=subsPot,
    chemPotDict=chemPotDict,
    zLim=zLim,
)

# --- 4. 재개(Restart) 로직 ---
try:
    list_file = os.listdir('.')
    list_job = [f for f in list_file if f[0] == 's' and '.' not in f and f != 'scratch']
    if 'sub-POSCAR' in list_job:
        list_job.remove('sub-POSCAR')
    list_jid = [int(j[1:]) for j in list_job]
    kidnum = max(list_jid)
    print(f'Restarting the search from kidnum = {kidnum}')
except Exception:
    kidnum = 0
    print('New search! Starting from kidnum = 0')

if kidnum == 0:
    print('Initializing the population with custom parents in gcga.db...')
    pop.initializeDB()
    pop.natural_selection()


def free_vram(*objs):
    """삭제 + 가비지 컬렉션 + CUDA 캐시 비우기."""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# --- 5. 메인 진화 루프 ---
cwd = os.getcwd()

pop.reinitializeDB()

CONSEC_OOM_LIMIT = 3   # 연속 OOM 임계치 → 도달 시 프로세스 재실행
consec_oom = 0

while not pop.is_converged() or kidnum < minConf:
    if 'STOP' in os.listdir():
        print('STOP REQUESTED BY USER')
        break
    if kidnum > totConf:
        print('MAX # SAMPLE REACHED')
        break

    kidnum += 1
    kiddir = f's{str(kidnum).zfill(6)}'
    try:
        os.mkdir(kiddir)
    except FileExistsError:
        pass
    os.chdir(kiddir)

    kid = None
    kid_opt = None
    print(f'\n--- Generating Child {kiddir} ---')

    try:
        while kid is None:
            kid = pop.gen_offspring_box(
                mutRate=0.5,
                xyzLims=xyzLims,
                transVec=[[-1, 1, -2, 2, -3, 3, -4, 4, -6, 6, -12, 12],
                          [-1, 1, -2, 2, -3, 3, -6, 6]]
            )

        kid.preopt_hooke(cutoff=1.2, toler=0.1)
        kid.write('POSCAR')

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
            free_vram(kid, kid_opt)
            continue

        f64 = np.asarray(kid_opt.get_forces(), dtype=np.float64)
        if kid_opt.calc is not None:
            kid_opt.calc.results["forces"] = f64

        os.chdir(cwd)

        pop.add_aseResult(kid_opt, workdir=kiddir)
        pop.natural_selection()

        print(f'-> {kiddir} 처리 완료 및 Population 갱신.')
        consec_oom = 0

    except torch.cuda.OutOfMemoryError as e:
        # OOM: 현재 자손 폐기, 메모리 비우고 다음으로
        print(f'⚠️ CUDA OOM on {kiddir}: {e}')
        os.chdir(cwd)
        free_vram(kid, kid_opt)
        consec_oom += 1
        if consec_oom >= CONSEC_OOM_LIMIT:
            print(f'❌ 연속 {consec_oom}회 OOM — 프로세스를 재실행하여 GPU 컨텍스트를 초기화합니다.')
            sys.stdout.flush()
            os.execv(sys.executable, [sys.executable, '-u'] + sys.argv)
        continue

    except Exception as e:
        # 그 외 예외: 로깅 후 다음 자손으로 (DB 일관성 유지를 위해 cwd 복귀 보장)
        print(f'⚠️ {kiddir}에서 예외 발생: {type(e).__name__}: {e}')
        traceback.print_exc()
        try:
            os.chdir(cwd)
        except Exception:
            pass
        free_vram(kid, kid_opt)
        continue

    finally:
        # 매 iteration 마다 VRAM 정리 (성공/실패 무관)
        free_vram()

    sleep(1)

if pop.is_converged():
    print('\n🎉 CONVERGED! GLOBAL MINIMUM FOUND!')
else:
    print('\n🏁 TERMINATED!')
