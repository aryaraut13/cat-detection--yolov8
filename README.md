# cat-detection--yolov8
# Cat Detection Using YOLOv8

## Overview

This project builds an object detection system that identifies cats in images using bounding boxes and confidence scores. The solution is built on YOLOv8, a fast and accurate single-stage detection architecture, trained via transfer learning on a labeled cat image dataset from Roboflow.

---

## Methodology and Approach

### Model Selection

YOLOv8 Nano (yolov8n) was chosen as the base model for the following reasons:

- It is a single-stage detector, meaning it predicts bounding boxes and class labels in one forward pass, making it significantly faster than two-stage models like Faster R-CNN
- Its pre-trained weights from the COCO dataset include the "cat" class, which gives the model a strong starting point through transfer learning
- The nano variant is lightweight enough to train on free cloud GPUs within a reasonable time while still producing competitive accuracy

### Transfer Learning Strategy

Rather than training from scratch, the model was initialized with COCO pre-trained weights and then fine-tuned on the Roboflow cat dataset. This approach requires far fewer epochs and less data to converge because the model already understands low-level visual features like edges, textures, and shapes.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | yolov8n.pt |
| Epochs | 30 |
| Image Size | 640 × 640 |
| Batch Size | 8 |
| Device | GPU (CUDA) |
| Optimizer | AdamW (default) |
| Early Stopping Patience | 15 epochs |
| Data Augmentation | Mosaic, horizontal flip, HSV jitter, scale |

---

## Dataset

- **Source:** Roboflow Universe — catnpklm dataset, Version 1
- **Format:** YOLOv8 (normalized bounding box annotations in .txt files)
- **Classes:** 1 (cat)
- **Annotation Format:** Each label file contains one line per object — `class_id center_x center_y width height` (all values normalized between 0 and 1)

| Split | Image Count |
|-------|-------------|
| Train | 139 |
| Validation | 43 |
| Test | 18 |

---

## Assumptions

1. All labeled objects in the dataset represent domestic cats and the annotations are accurate.
2. Images containing no annotation file, or an empty label file, are treated as negative samples — no cats present.
3. A confidence threshold of 0.5 is applied during inference, meaning any detection below 50% confidence is discarded.
4. An IoU (Intersection over Union) threshold of 0.45 is used during Non-Maximum Suppression to remove overlapping duplicate detections.
5. Transfer learning from COCO weights is valid for this task because cat is one of the 80 original COCO classes.
6. The validation set provided by Roboflow is representative of real-world cat images the model will encounter.

---

## Model Performance and Results

Evaluation was performed on the validation split using the trained best.pt weights.

| Metric | Score |
|--------|-------|
| Precision | 0.6258  |
| Recall | 0.6200  |
| mAP@0.5 | 0.6325  |
| mAP@0.5:0.95 | 0.2819 |

**What these numbers mean:**

- **Precision** measures how many of the model's detections were actually correct. A higher value means fewer false positives.
- **Recall** measures how many of the real cats in images were successfully detected. A higher value means fewer missed detections.
- **mAP@0.5** is the primary benchmark metric — it measures average precision across all images at a 50% overlap threshold between predicted and ground truth boxes.
- **mAP@0.5:0.95** is a stricter version averaged across multiple overlap thresholds from 0.5 to 0.95.

---

## Project Structure

```
cat_detection_submission/
├── CatDetection.ipynb          # Complete notebook with all code and outputs
├── best.pt                     # Trained model weights
├── README.md                   # This document
├── inference_results.png       # Sample predictions with bounding boxes
├── sample_annotations.png      # Ground truth visualization from training data
└── evaluation_results.txt      # Saved precision, recall, and mAP scores
```



## Challenges Encountered

1. Dataset size limitation
The dataset contains a limited number of images which increases the risk of overfitting. This was mitigated by relying on pre-trained weights and using built-in data augmentation during training.

2. Variable object scales
Cats appear at very different scales across images — some fill the frame, others are small and distant. YOLOv8's multi-scale detection head (P3, P4, P5) handles this inherently without any extra configuration.

3. Runtime disconnections on Google Colab
Colab's free tier disconnects after periods of inactivity. This required re-running import cells before continuing. All critical imports were added to the top of the training cell to make re-runs self-contained.

4. Nested output directory
An incorrect `project` parameter in the training call caused outputs to save into a nested path (`runs/detect/runs/detect/cat_detector/`) rather than the expected location. This was resolved by searching recursively for `best.pt` and using its actual path.



Suggestions for Future Improvements

1. Use a larger model variant
Upgrading from YOLOv8n to YOLOv8s or YOLOv8m would improve accuracy at the cost of slightly longer inference time. This trade-off is acceptable if deployed on a server rather than an edge device.

2. Expand the dataset
Combining this dataset with other cat detection datasets on Roboflow Universe would expose the model to more variation in lighting, backgrounds, and cat breeds, improving generalization.

3. Hyperparameter tuning
The Ultralytics library supports automated hyperparameter search via `model.tune()`. Running this on a larger compute budget could meaningfully improve mAP.

4. Deploy as a REST API
The trained model can be wrapped in a FastAPI endpoint, allowing other applications or Power Automate workflows to send images and receive detection results — directly relevant to the AI/ML automation role.

5. Real-time video detection
YOLOv8 supports video inference with built-in object tracking (`model.track()`), which would enable real-time cat detection from webcam or CCTV feeds.

6. Quantization for edge deployment
The model can be exported to ONNX or TensorRT format for faster inference on devices with limited resources, useful for embedded or mobile applications.



## Dependencies

ultralytics
roboflow
opencv-python
matplotlib
Pillow
torch
torchvision
numpy
PyYAML
