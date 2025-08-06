from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8

import os.path as osp

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
classes = ['left robot hand', 'right robot hand']
base_model = GroundedSAM(ontology=CaptionOntology({"left robot hand" : "left robot hand", "right robot hand" : "right robot hand"}))
root_path = "/home/scilab/Documents/teleoperation"


# label all images in a folder called `context_images`
base_model.label(
  input_folder=osp.join(root_path,"robot_hand_img"),
  output_folder=osp.join(root_path, "output_robot_hand")
)