import os
import random
import glob
import numpy as np
from ase.io import read, write
from ase.geometry import find_mic

# -----------------------------------------------------------------------------
# 하이퍼파라미터 (HYPERPARAMETERS)
# -----------------------------------------------------------------------------
seed = 42

NUM_STRUCTURES = 50           # 총 생성할 slab-cluster 구조 개수
NUM_CLUSTERS_PER_SLAB = 8     # 각 슬랩 위에 올릴 클러스터의 개수 (8 x 32 = 256 Atoms)

# 파일 및 디렉토리 경로 설정
SLAB_PATH = "/home/kyunghun/05_AGOX/13/00_slab/4_Ag_slab_supercell_opt.vasp"
CLUSTER_DIR = "/home/kyunghun/05_AGOX/13/01_cluster/Ag_32_relaxed"
OUTPUT_DIR = "./generated_surfaces"

# 배치 알고리즘 설정
MIN_DISTANCE_GAP = 3.0  # 클러스터 외접원 사이의 최소 간격 (Angstrom)
MAX_ATTEMPTS = 5000     # 무작위 위치 생성을 시도할 최대 횟수
# -----------------------------------------------------------------------------

def process_cluster(atoms):
    """
    클러스터를 원점으로 정렬하고 외접원의 반경을 계산하는 함수.
    1. XY 평면의 중심을 (0, 0)으로 이동.
    2. 가장 낮은 원자의 Z 좌표를 0으로 설정.
    3. XY 평면에서의 최대 반경 계산.
    """
    positions = atoms.get_positions()
    
    # 1. XY 평면 무게중심 계산 및 이동
    center_xy = np.mean(positions[:, :2], axis=0)
    positions[:, :2] -= center_xy
    
    # 2. Z축 높이 정렬 (가장 낮은 곳이 0)
    min_z = np.min(positions[:, 2])
    positions[:, 2] -= min_z
    
    atoms.set_positions(positions)
    
    # 3. 중심(0,0)으로부터 XY 평면상의 최대 거리(반경) 계산
    distances_xy = np.linalg.norm(positions[:, :2], axis=1)
    radius = np.max(distances_xy)
    
    return atoms, radius

def main():
    print("==================================================")
    print("🚀 8개 Ag32 클러스터 무작위 배치 및 50개 슬랩 생성 시작")
    print("==================================================")
    print(f"목표 구조 개수    : {NUM_STRUCTURES}개")
    print(f"슬랩당 클러스터 수: {NUM_CLUSTERS_PER_SLAB}개")
    print(f"최소 틈새 간격    : {MIN_DISTANCE_GAP} Å")
    
    # 1. 슬랩(Slab) 정보 로드
    if not os.path.exists(SLAB_PATH):
        raise FileNotFoundError(f"슬랩(Slab) 파일을 찾을 수 없습니다: {SLAB_PATH}")
    
    slab = read(SLAB_PATH)
    cell = slab.get_cell()
    z_max = np.max(slab.get_positions()[:, 2])
    z_placement = z_max + 2.5 # 슬랩 상단으로부터 2.5Å 위에 배치
    
    # 2. 최적화된 클러스터 리스트 확인
    cluster_files = sorted(glob.glob(os.path.join(CLUSTER_DIR, "relaxed_cluster_*.vasp")))
    if len(cluster_files) < NUM_CLUSTERS_PER_SLAB:
        raise ValueError(
            f"클러스터 개수가 부족합니다. 최소 {NUM_CLUSTERS_PER_SLAB}개가 필요하지만, "
            f"'{CLUSTER_DIR}' 폴더에 {len(cluster_files)}개만 존재합니다."
        )
        
    print(f"로드할 수 있는 최적화 클러스터 총 개수: {len(cluster_files)}개")
    
    # 3. 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 4. 50개 구조 생성 루프
    for idx in range(NUM_STRUCTURES):
        # 각 구조마다 독립적이면서도 재현 가능한 고유 시드 부여
        struct_seed = seed + idx
        random.seed(struct_seed)
        np.random.seed(struct_seed)
        
        print(f"\n[{idx+1:02d}/{NUM_STRUCTURES:02d}] 구조 생성 중 (Seed: {struct_seed})...")
        
        success = False
        retry_count = 0
        
        # 만약 무작위 배치 공간이 부족하여 막히면 시드를 변경하며 다시 배치
        while not success and retry_count < 10:
            # 100개 클러스터 중 중복 없이 8개 선택
            chosen_files = random.sample(cluster_files, NUM_CLUSTERS_PER_SLAB)
            
            # 클러스터 구조 로드 및 중심정렬/반경 계산
            clusters = []
            radii = []
            for file_path in chosen_files:
                atoms = read(file_path)
                proc_atoms, r = process_cluster(atoms)
                clusters.append(proc_atoms)
                radii.append(r)
            
            placed_info = [] # [(Cartesian 위치, 반경), ...]
            final_atoms = slab.copy()
            
            all_placed = True
            
            # 8개 클러스터 배치 시도
            for i in range(NUM_CLUSTERS_PER_SLAB):
                cluster = clusters[i]
                r = radii[i]
                
                placed = False
                for attempt in range(MAX_ATTEMPTS):
                    # 분수 좌표(0~1) 생성 후 Cartesian 좌표로 변환
                    f_pos = np.array([random.uniform(0, 1), random.uniform(0, 1), 0.0])
                    c_pos = np.dot(f_pos, cell)
                    
                    # 충돌 및 중첩 검사 (주기적 경계 조건 PBC 고려)
                    collision = False
                    for p_pos, p_r in placed_info:
                        diff = c_pos - p_pos
                        mic_vec, mic_dist = find_mic(diff, cell, pbc=True)
                        
                        # XY 평면상의 거리 추출
                        mic_vec = mic_vec.flatten()
                        dist_xy = np.linalg.norm(mic_vec[:2])
                        
                        # (두 클러스터의 외접원 반경 합 + 안전 틈새 거리)보다 가깝다면 충돌로 간주
                        if dist_xy < (r + p_r + MIN_DISTANCE_GAP):
                            collision = True
                            break
                    
                    # 충돌이 없으면 배치
                    if not collision:
                        placed_info.append((c_pos, r))
                        
                        # 클러스터 좌표 이동 (XY 평면은 배치 좌표로, Z축은 슬랩 높이에 맞게)
                        cluster_pos = cluster.get_positions()
                        cluster_pos[:, 0] += c_pos[0]
                        cluster_pos[:, 1] += c_pos[1]
                        cluster_pos[:, 2] += z_placement
                        cluster.set_positions(cluster_pos)
                        
                        # 슬랩 시스템에 클러스터 병합
                        final_atoms += cluster
                        placed = True
                        break
                
                # 만약 한 클러스터라도 배치가 불가능한 경우 루프 탈출 후 전체 재시도
                if not placed:
                    all_placed = False
                    print(f"   ⚠️ 클러스터 {i+1} 배치 공간 확보 실패. 시드를 조정한 후 재시도합니다.")
                    break
            
            if all_placed:
                output_filename = os.path.join(OUTPUT_DIR, f"surface_{idx:02d}.vasp")
                write(output_filename, final_atoms, format='vasp')
                print(f"   ✅ 구조 생성 완료: {output_filename}")
                success = True
            else:
                retry_count += 1
                # 시드 재조정
                random.seed(struct_seed * 100 + retry_count)
                np.random.seed(struct_seed * 100 + retry_count)
                
        if not success:
            print(f"❌ 에러: 구조 {idx} 생성 실패 (배치 공간 부족). 슬랩 크기를 늘리거나 간격을 줄여야 합니다.")
            return

    print("\n" + "="*50)
    print(f"🎉 성공! 총 {NUM_STRUCTURES}개의 고유 슬랩 구조가 '{OUTPUT_DIR}/' 폴더 내에 생성되었습니다.")
    print("==================================================")

if __name__ == "__main__":
    main()
