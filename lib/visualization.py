import os, sys
import time
import numpy as np
import open3d as o3d
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from typing import Union, Literal
import glob
import copy
from lib import tools

def viz_img_bbox(
        gt_data:dict,
        imgs_data:dict,
        save_path:str, 
        **kwargs
        ) -> None:
        """Visualize bbox and image using OpenCV

        Args:
            - gt_data (dict): Dictionary containing ground truth data
            - imgs_data (dict): Dictionary containing the images ids, image path, image data(np) ...
            - save_path (str, optional): path for saving the figure". Defaults to None.
        """

        print(
                "=====================Keyboard Shortcuts==================",
                "A-> Previous Image",
                "D-> Next Image",
                "S-> Save image",
                "G-> Toggle Ground Truth",
                "Q/Esc-> Quite", 
                "H-> Help",
                "============================End==========================",
                sep="\n"
            )
        
        # TODO Visualize the 3d bbox

        imgs_id = list(imgs_data.keys())

        # Params
        color_gt= [255, 0, 0]
        current_idx = 0  # Start at the first image
        show_gt=True
        while True:
            img_id = imgs_id[current_idx]

            # Get image data
            img = copy.copy(imgs_data[img_id]['img_data'])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Draw bounding boxes and display the class for GT
            if show_gt and img_id in gt_data:
                
                # Frame data ground truth
                gt_bboxes = gt_data[img_id]['bboxes']

                # Skip if there is not prediction
                if len(gt_bboxes) > 0:
                    for gt_bbox in gt_bboxes:  
                        x1, y1, x2, y2 = map(lambda x:int(x), gt_bbox['pos_xyxy'])  # BBox corner points
                        
                        # Draw the bounding box
                        cv2.rectangle(
                            img=img, 
                            pt1=(x1, y1), 
                            pt2=(x2, y2), 
                            color=color_gt, 
                            thickness=1
                        )
                        
                        # Add label on the image
                        cv2.putText(
                            img=img, 
                            text=f"Cls: {gt_bbox['cls_name']}", 
                            org=(x1, y1 - 10), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.5, 
                            color=color_gt, 
                            thickness=1
                        )
                else:
                    print(f"[WARN] No Ground truth data for image '{img_id}'")

            # Add labels for Prediction and Ground Truth
            cv2.putText(img, 'Ground Truth', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_gt, 2)
            
            # Show the image
            cv2.imshow('Detection', img)
            
            # Wait for the user to press a key
            key = cv2.waitKey(0)  # Wait indefinitely until a key is pressed
            
            # print(f"Key pressed is: {key}") # check the key press
            
            if key in [113, 27]:  # Q or ESC key to exit
                print("Exiting...")
                break

            elif key == 103: # G key to show GT
                show_gt = not show_gt
                print(f"GT visualization: {show_gt}")

            elif key == 97:  # A key --> previous image (if not first image)
                if current_idx > 0:
                    current_idx -= 1  # Decrease index
                else:
                    print("You are at the first image.")

            elif key == 100:  # D key --> next image (default behavior)
                if current_idx < len(imgs_id) - 1:
                    current_idx += 1  # Increase index
                else:
                    print("You are at the last image.")

            elif key == 115: # S key to save the image
                save_path_ = os.path.join(save_path, img_id+'.png')
                cv2.imwrite(save_path_, img)
                print(f"Saved image to: {save_path_}")
            
            elif key == 104: # H key for help
                print(
                        "=====================Keyboard Shortcuts==================",
                        "A-> Previous Image",
                        "D-> Next Image",
                        "S-> Save image",
                        "P-> Toggle Predictions",
                        "Q/Esc-> Quite", 
                        "H-> Help",
                        "============================End==========================",
                        sep="\n"
                    )

        # Close all OpenCV windows
        cv2.destroyAllWindows()
