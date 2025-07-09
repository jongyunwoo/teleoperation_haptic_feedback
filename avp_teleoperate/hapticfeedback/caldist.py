from autodistill_clip import CLIP
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
import supervision as sv

from autodistill.core.composed_detection_model import ComposedDetectionModel
import cv2
import numpy as np
classes = ["robot hand"]

SAMCLIP = ComposedDetectionModel(
    detection_model=GroundedSAM(
        CaptionOntology({"robot hand": "robot_hand"})
    ),
    classification_model=CLIP(
        CaptionOntology({k: k for k in classes})
    )
)
def caldist(image, depth, depth_scale):
    
    depth_scale = depth_scale
    image_path = image
    depth_raw = cv2.imread(depth, cv2.IMREAD_UNCHANGED)
    
    results = SAMCLIP.predict(image_path)

    image = cv2.imread(image_path)

    annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()
    masks = results.mask
    # labels = [
    #     f"{classes[class_id]} {confidence:0.2f}"
    #     for _, _, confidence, class_id, _ in results
    # ]
    
    labels = [
        f"{classes[int(cid)]} {conf:.2f}"
        for conf, cid in zip(results.confidence, results.class_id)
    ]

    combined_mask = np.any(masks, axis=0)
    ys, xs = np.where(combined_mask)
    min_idx = np.argmin(ys)
    top_y = ys[min_idx]
    top_x = xs[min_idx]
    
    d = depth_raw[top_x, top_y]
    depth_m = d * depth_scale
    
    return depth_m
    
    # print(f"x={top_x}, y = {top_y}")
    # annotated_frame = annotator.annotate(
    #     scene=image.copy(), detections=results
    # )
    # annotated_frame = label_annotator.annotate(
    #     scene=annotated_frame, labels=labels, detections=results
    # )

    # sv.plot_image(annotated_frame, size=(8, 8))