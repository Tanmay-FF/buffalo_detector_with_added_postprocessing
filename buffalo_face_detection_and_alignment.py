#!/usr/bin/env python3

"""
This script runs a face detection and alignment pipeline on a single image
using a pre-built ONNX model with embedded post-processing.

It takes an image path and a device (cpu/cuda) as input, detects faces,
and for each face, it saves a cropped version and an aligned version.
It also outputs a JSON file with metadata about the detections.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from skimage import transform as trans
import warnings
import json
import argparse
import sys       

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Global Settings & Constants ---

# Suppress unnecessary warnings
ort.set_default_logger_severity(3)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Default thresholds for the NMS operator
# Note: These are passed to the model at runtime.
CONF_THRESH = 0.4
NMS_THRESH = 0.4

# Output size for the aligned face images
ALIGNMENT_SIZE = 224

# Standard 5-point landmark destination for ArcFace alignment
ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

# --- Pre-processing Function ---

def preprocess(img, input_size=(640, 640)):
    """
    Preprocesses the image for the SCRFD ONNX model.

    This involves:
    1. Resizing the image to fit 640x640 while maintaining aspect ratio.
    2. Padding the image to 640x640.
    3. Normalizing the image (0-255 -> -1.0 to 1.0).
    4. Transposing dimensions to NCHW format.

    Args:
        img (np.ndarray): The input image (BGR, HWC).
        input_size (tuple): The target input size (Width, Height).

    Returns:
        np.ndarray: The processed blob (NCHW, float32).
        float: The scaling factor used to resize the image.
    """
    
    h_orig, w_orig = img.shape[:2]
    
    # Calculate new size maintaining aspect ratio
    im_ratio = float(h_orig) / w_orig
    model_ratio = float(input_size[1]) / input_size[0]  # H/W
    
    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    
    # This scale is used to map detections back to the original image size
    det_scale = float(new_height) / h_orig
    
    resized_img = cv2.resize(img, (new_width, new_height))
    
    # Create a 640x640 canvas and paste the resized image
    det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized_img
    
    # --- Normalization and Formatting ---
    det_img_rgb = det_img[..., ::-1]  # BGR to RGB
    
    # Normalize from [0, 255] to [-1.0, 1.0]
    blob = (det_img_rgb.astype(np.float32) - 127.5) / 128.0
    
    # Transpose from HWC to CHW
    blob = blob.transpose(2, 0, 1)
    
    # Add batch dimension (NCHW)
    blob = np.expand_dims(blob, axis=0)
    
    return blob, det_scale

# --- ONNX and Alignment Functions ---

def init_session(onnx_model_path, device):
    """
    Initialize an ONNX Runtime inference session.

    Args:
        onnx_model_path (str): Path to the ONNX model file.
        device (str): "cuda" or "cpu".

    Returns:
        ort.InferenceSession: The initialized session.
    """
    if device.lower() == "cuda":
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    return session

def estimate_norm(lmk, image_size=224):
    """
    Estimate the similarity transformation matrix for face alignment.

    Args:
        lmk (np.ndarray): The 5 facial landmarks [5, 2].
        image_size (int): The target output size.

    Returns:
        np.ndarray: The 2x3 transformation matrix.
    """
    # Adjust ArcFace destination points based on the target image size
    ratio = image_size / 112.0 if image_size % 112 == 0 else image_size / 128.0
    diff_x = 0 if image_size % 112 == 0 else 8.0 * ratio
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    
    # Estimate the transformation
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    return tform.params[0:2, :]

def align_and_crop(img, landmark, image_size=224):
    """
    Align and crop a face using the estimated transformation.

    Args:
        img (np.ndarray): The original full image.
        landmark (np.ndarray): The 5 facial landmarks [5, 2].
        image_size (int): The target output size.

    Returns:
        np.ndarray: The aligned and cropped face image.
    """
    M = estimate_norm(landmark, image_size)
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

# --- Core Processing Functions ---

def load_detector(device):
    """
    Load the Buffalo face detector ONNX model.

    Args:
        device (str): "cuda" or "cpu".

    Returns:
        ort.InferenceSession: The initialized detector session.
    """
    model_path = "models/scfrd_640_with_postprocessing_and_dynamic_thresholding.onnx"
    if not os.path.exists(model_path):
        print(f"ERROR: Buffalo model not found at: {model_path}.")
        print("Please ensure your new model (with post-processing) is at this location.")
        sys.exit(1)
        
    sess = init_session(model_path, device)
    print(f"Buffalo detector (with post-processing) loaded successfully on {device.upper()}")
    return sess

def detect_faces(sess, img, device):
    """
    Detect faces in an image using the loaded detector.

    Args:
        sess (ort.InferenceSession): The ONNX detector session.
        img (np.ndarray): The input image (BGR, HWC).
        device (str): "cuda" or "cpu".

    Returns:
        list[dict]: A list of detection dictionaries.
    """
    print("Performing detection...")
    
    # 1. Preprocess the image
    blob, det_scale = preprocess(img, input_size=(640, 640))
    
    # 2. Define dynamic NMS thresholds
    # These must be numpy arrays of type float32.
    score_thresh_val = np.array([CONF_THRESH], dtype=np.float32)
    iou_thresh_val = np.array([NMS_THRESH], dtype=np.float32)

    # 3. Get input names from the model
    # The model has 3 inputs: 'input', 'iou_threshold_input', 'score_threshold_input'
    input_names = [i.name for i in sess.get_inputs()]
    main_input_name = input_names[0]  # Assumes 'input' is the first one
    
    # 4. Run inference
    outputs = sess.run(None, {
        main_input_name: blob,
        "score_threshold_input": score_thresh_val,
        "iou_threshold_input": iou_thresh_val
    })
    
    # 5. Unpack outputs
    # These are the final NMS-filtered results
    boxes, scores, kps = outputs[0], outputs[1], outputs[2]

    if boxes.shape[0] == 0:
        print("No faces found.")
        return []
        
    # 6. Post-process: Scale results back to original image size
    boxes /= det_scale
    kps /= det_scale
    
    # Reshape keypoints from [N, 10] to [N, 5, 2]
    kps = kps.reshape((kps.shape[0], -1, 2))
    
    # 7. Format results
    detections = []
    for i in range(boxes.shape[0]):
        det = {
            "bbox": boxes[i],       # [x1, y1, x2, y2]
            "kps": kps[i],          # [5, 2]
            "det_score": scores[i]  # float
        }
        detections.append(det)
    
    return detections

def process_single_image(img_path, detector_sess, device):
    """
    Full pipeline for a single image:
    Detect -> Crop -> Align -> Save Results.

    Args:
        img_path (str): Path to the input image.
        detector_sess (ort.InferenceSession): The ONNX detector session.
        device (str): "cuda" or "cpu".
    """
    print(f"\nProcessing image: {img_path}")
    
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not read image: {img_path}")
        return
    print(f"Image loaded successfully - Shape: {img.shape}")

    # Create a dedicated output directory for this image
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = f"output_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    # Save a copy of the original image
    original_path = os.path.join(output_dir, f"{base_name}_original.jpg")
    cv2.imwrite(original_path, img)
    print(f"Original image saved: {original_path}")

    # --- Run Detection ---
    print("\nStarting face detection...")
    detections = detect_faces(detector_sess, img, device)
    
    if not detections:
        print("No faces detected in the image.")
        return
    
    print(f"Successfully detected {len(detections)} face(s)")
    
    detection_info = []
    
    # --- Process Each Detection ---
    for idx, det in enumerate(detections):
        print(f"\nProcessing detected face {idx + 1}/{len(detections)}")
        
        bbox = det["bbox"]
        landmarks = det["kps"]
        det_score = det["det_score"]
              
        # --- Save Cropped Face ---
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within image bounds
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(img.shape[0], y2)
        x2 = min(img.shape[1], x2)
        
        cropped_face = img[y1:y2, x1:x2]
        
        if cropped_face.size == 0:
            print(f"Warning: Cropped face {idx} is empty (invalid bbox). Skipping save.")
            continue
            
        cropped_path = os.path.join(output_dir, f"face_{idx}_cropped.jpg")
        cv2.imwrite(cropped_path, cropped_face)
        print(f"Cropped detection saved: {cropped_path}")
        
        # --- Save Aligned Face ---
        print("Performing face alignment...")
        aligned_img = align_and_crop(img, landmarks, ALIGNMENT_SIZE)
        
        aligned_path = os.path.join(output_dir, f"face_{idx}_aligned_{ALIGNMENT_SIZE}x{ALIGNMENT_SIZE}.jpg")
        cv2.imwrite(aligned_path, aligned_img)
        print(f"Aligned face saved: {aligned_path}")
        
        # --- Store Metadata ---
        face_info = {
            "face_idx": idx,
            "bbox": bbox.tolist(),
            "landmarks": landmarks.tolist(),
            "det_score": float(det_score),
            "detection_output": cropped_path,
            "aligned_path": aligned_path
        }
        detection_info.append(face_info)
    
    # --- Save Final Metadata JSON ---
    print("\nSaving metadata...")
    metadata_path = os.path.join(output_dir, "detection_results.json")
    metadata = {
        "image_path": img_path,
        "image_shape": img.shape,
        "device": device,
        "alignment_size": ALIGNMENT_SIZE,
        "num_faces_detected": len(detections),
        "faces": detection_info
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")
    print(f"Processing complete for {img_path}")

# --- Main Execution ---

def main():
    """
    Main processing pipeline.
    Parses command-line arguments, loads the model, and processes the image.
    """
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Run face detection and alignment on a single image.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the input image file."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (e.g., 'cpu' or 'cuda'). \nDefault: 'cpu'"
    )
    
    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.exists(args.image_path):
        print(f"ERROR: Image file not found at: {args.image_path}")
        sys.exit(1)

    # --- Load Model and Process ---
    try:
        detector_sess = load_detector(args.device)
        process_single_image(args.image_path, detector_sess, args.device)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()