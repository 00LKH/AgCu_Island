import os
from ase.io import read, write

# 1. 경로 및 폴더 설정
traj_path = "all_candidates_Ag32.traj"
output_dir = "Ag_32"

# 2. 저장할 폴더 생성 (이미 존재하면 건너뜀)
os.makedirs(output_dir, exist_ok=True)

# 3. Trajectory 파일로부터 모든 클러스터 구조 로드
try:
    clusters = read(traj_path, index=":")
    print(f"✅ 성공적으로 {traj_path} 파일을 로드했습니다.")
    print(f"   - 총 클러스터 개수: {len(clusters)}개")
    print(f"   - 화학식: {clusters[0].get_chemical_formula()}")
except Exception as e:
    print(f"❌ Trajectory 파일을 로드하는 중 오류가 발생했습니다: {e}")
    clusters = []

# 4. 각 클러스터를 .vasp 파일로 저장
if len(clusters) > 0:
    for i, cluster in enumerate(clusters):
        # 파일 이름을 3자리 패딩(cluster_000.vasp)으로 정렬하기 쉽게 지정
        vasp_filename = os.path.join(output_dir, f"cluster_{i:03d}.vasp")
        
        # 격자(Cell)가 정상적인지 확인 후 VASP 포맷으로 저장
        write(vasp_filename, cluster, format="vasp")
        
    print(f"\n🎉 변환 완료! 모든 클러스터 파일이 '{output_dir}/' 폴더 내에 저장되었습니다.")
