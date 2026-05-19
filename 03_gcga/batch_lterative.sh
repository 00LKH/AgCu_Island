#!/bin/bash
#PBS -N gcga_safe
#PBS -q full
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -j oe

source /home/kyunghun/anaconda3/etc/profile.d/conda.sh
conda activate agox_uma
export PYTHONPATH=$PYTHONPATH:/home/kyunghun/08_GOCIA/gocia

# cd $PBS_O_WORKDIR
# python -u 00_prep_test_trajectory.py

cd $PBS_O_WORKDIR
# --- 체인 종료 조건 ---
# DONE: python이 정상 종료(수렴/max sample/STOP) 시 자동 생성
# STOP_CHAIN: 사용자가 직접 만들어 즉시 체인 중단 가능 (python loop의 STOP과 분리)
if [ -f DONE ]; then
    echo "[$(date)] DONE 파일 존재 — 체인 종료."
    exit 0
fi
if [ -f STOP_CHAIN ]; then
    echo "[$(date)] STOP_CHAIN 파일 존재 — 체인 종료. (재개하려면 STOP_CHAIN 삭제)"
    exit 0
fi

# --- 다음 job을 먼저 큐에 넣어둠 (afterany: 성공/실패/walltime/OOM kill 모두 포함) ---
# 미리 큐잉해야 현재 job이 hard crash 되어도 후속 job이 살아남음
NEXT_JID=$(qsub -W depend=afterany:$PBS_JOBID batch_safe.sh)
QSUB_RC=$?
if [ $QSUB_RC -eq 0 ]; then
    echo "[$(date)] 다음 job 예약됨: $NEXT_JID (afterany:$PBS_JOBID)"
else
    echo "[$(date)] ⚠️  다음 job 예약 실패 (rc=$QSUB_RC). 이 job만 실행됨."
fi

# --- 실제 계산 ---
python -u 02_gcga_run_safe.py
EXIT_CODE=$?

# 정상 종료(exit 0) 시 DONE 생성 → 큐에 대기 중인 다음 job은 위 가드에서 즉시 종료
# 비정상 종료(OOM kill, walltime, segfault 등) 시 DONE 안 만듦 → 다음 job이 이어받아 재개
if [ $EXIT_CODE -eq 0 ]; then
    touch DONE
    echo "[$(date)] python exit 0 — DONE 생성, 체인 종료 예정."
else
    echo "[$(date)] python 비정상 종료 (exit=$EXIT_CODE) — 다음 job이 재개."
fi

exit $EXIT_CODE
