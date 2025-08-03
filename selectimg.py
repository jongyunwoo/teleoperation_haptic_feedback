import os
import random
import shutil
import time

source_folder = "/home/scilab/Documents/teleoperation/dataset/images"
destination_folder = "/home/scilab/Documents/teleoperation/dataset/valid"

# 이미지 파일 목록 가져오기
all_images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 200장 랜덤 추출 (중복 허용)
selected_images = random.choices(all_images, k=100)

# output 폴더 없으면 생성
os.makedirs(destination_folder, exist_ok=True)

# 이미지 복사 (항상 새로운 파일명으로 저장)
for img in selected_images:
    src_path = os.path.join(source_folder, img)
    # 고유 파일명 생성: 타임스탬프 + 랜덤값 + 원본 확장자
    ext = os.path.splitext(img)[1]
    unique_name = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}{ext}"
    dst_path = os.path.join(destination_folder, unique_name)
    shutil.copy(src_path, dst_path)
    time.sleep(0.001)  # 타임스탬프 중복 방지용 약간의 delay

print("✅ 중복 이미지도 모두 새로운 파일명으로 저장 완료!")
