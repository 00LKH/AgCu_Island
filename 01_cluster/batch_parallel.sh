#!/bin/bash
#PBS -N rss_parallel
#PBS -q full
#PBS -l select=1:ncpus=20:ngpus=1
#PBS -j oe

source /home/kyunghun/anaconda3/etc/profile.d/conda.sh
conda activate agox_uma

cd $PBS_O_WORKDIR

# GPU 메모리를 고려하여 동시에 실행할 프로세스 수 (10~20% 이므로 5~8개 적당)
MAX_JOBS=1

for n_ag in {32..32}
do
    # 백그라운드(&)로 실행하고, 각 출력은 개별 로그 파일에 저장
    python -u rss_parallel.py --n_ag $n_ag > log_Ag${n_ag}.out 2>&1 &
    
    # 현재 실행 중인 백그라운드 작업(jobs)의 개수가 MAX_JOBS 이상이면 대기
    while [ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]; do
        sleep 5
    done
done

# 모든 백그라운드 작업이 종료될 때까지 대기
wait
echo "All parallel jobs finished."

cd $PBS_O_WORKDIR
python generate_cluster.py

cd $PBS_O_WORKDIR
python cluster_relaxation.py

exit 0
