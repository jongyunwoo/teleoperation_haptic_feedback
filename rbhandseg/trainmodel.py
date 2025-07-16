# ─── 이 부분을 파일 최상단에 ───────────────────────────────────────────────
import cv2
from pathlib import Path
_orig_imread = cv2.imread
def _imread_pathfix(path, flags=cv2.IMREAD_COLOR):
    if isinstance(path, Path):
        path = str(path)
    return _orig_imread(path, flags)
cv2.imread = _imread_pathfix
# ──────────────────────────────────────────────────────────────────────────

import os
from autodistill.core.composed_detection_model import ComposedDetectionModel
from autodistill_clip import CLIP
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from autodistill_yolov8 import YOLOv8
import supervision as sv
# 
HOME = os.getcwd()
IMAGE_DIR_PATH   = f"{HOME}/images"
DATASET_DIR_PATH = os.path.join(HOME, "dataset")

classes = ["left robot hand", "right robot hand", 'left robot arm', 'right robot arm']
ontology = CaptionOntology({"left robot hand": "left robot hand",
                            "right robot hand" : "right robot hand",
                            'left robot arm' :'left robot arm',
                            'right robot arm' : 'right robot arm'})

base_model = GroundedSAM(ontology=ontology)

# label all images in a folder called `context_images`
base_model.label(
  input_folder="./images",
  output_folder="./dataset"
)

# # ─── 여기서 extension=".jpg" 로 반드시 마침표 포함 ─────────────────────────
dataset = base_model.label(
    input_folder="/home/scilab/Documents/teleoperation/images",
    # extension=".jpg",
    output_folder="/home/scilab/Documents/teleoperation/dataset"
)
# # ──────────────────────────────────────────────────────────────────────────
SAMPLE_SIZE = 16
SAMPLE_GRID_SIZE = (4, 4)
SAMPLE_PLOT_SIZE = (16, 10)
mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
images = []
image_names = []
for i, (image_path, image, annotation) in enumerate(dataset):
    if i == SAMPLE_SIZE:
        break
    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(
        scene=annotated_image, detections=annotation)
    annotated_image = box_annotator.annotate(
        scene=annotated_image, detections=annotation)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=annotation)

    image_names.append(Path(image_path).name)
    images.append(annotated_image)
    
sv.plot_images_grid(
    images=images,
    titles=image_names,
    grid_size=SAMPLE_GRID_SIZE,    size=SAMPLE_PLOT_SIZE)
print("✅ 라벨링 완료, 데이터셋은", DATASET_DIR_PATH, "에 생성되었습니다.")

target_model = YOLOv8("yolov8n-seg.pt")
target_model.train("./dataset/data.yaml", epochs=50, device="0")


