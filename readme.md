
# ONNX Face Detection and Alignment Pipeline

This project provides a complete pipeline for high-performance face detection and alignment using ONNX models. It features two main components:

1.  A utility script to embed all necessary post-processing (like anchor decoding and Non-Max Suppression) directly into a raw SCRFD/Buffalo-style ONNX face detector.
2.  A main inference script that uses the processed model to detect faces, extract 5-point landmarks, and save aligned, cropped face images.

## 🚀 Features

  * **High-Performance Inference:** Leverages `onnxruntime` for fast CPU or GPU-based detection.
  * **Embedded Post-processing:** The `add_postprocess_to_onnx.py` script bakes decoding and NMS logic directly into the ONNX graph. This simplifies deployment and eliminates the need for complex post-processing code in your application.
  * **Accurate Face Alignment:** Performs 5-point landmark detection and uses `skimage` to warp faces to a standard 224x224 ArcFace alignment.
  * **Comprehensive Output:** For each input image, the script saves:
      * Cropped bounding boxes for each detected face.
      * Aligned 224x224 images for each face.
      * A detailed `detection_results.json` file with coordinates, scores, and file paths.
  * **Dynamic Thresholds:** The modified ONNX model accepts dynamic score and IoU thresholds as runtime inputs, giving you flexible control over detection sensitivity.

## 📦 Project Structure

We recommend the following directory structure for clarity. The helper scripts are configured to look for models in a `models/` directory.

```
buffalo_detector_with_added_postprocessing/
├── models/
│   ├── buffalo_detector.onnx                                       # (INPUT) Your original, raw ONNX model
│   └── scfrd_640_with_postprocessing_and_dynamic_thresholding.onnx # (OUTPUT) The new model with post-processing
│
├── add_postprocess_to_onnx.py                  # Script to create the new model
├── buffalo_face_detection_and_alignment.py     # Script to run detection
├── requirements.txt                            # Project dependencies
└── README.md                                   # Readme file
```

## 🔧 Setup and Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Tanmay-FF/buffalo_detector_with_added_postprocessing.git
    cd buffalo_detector_with_added_postprocessing
    ```

2.  **Create a Virtual Environment (Recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    Install all required Python packages using the provided `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Get the Base Model**

      * You must obtain the original, **raw ONNX face detector model** (e.g., `buffalo_detector.onnx`). This model should have multiple raw outputs (for scores, boxes, and keypoints at different FPN levels).
      * Create a `models/` directory in your project root.
      * Place your downloaded raw model inside the `models/` directory.

## 🛠️ Usage (Quick Start)

The pipeline is a two-step process. You only need to run **Step 1** once to create your new model.

### Step 1: Add Post-processing to the Model

Run the `add_postprocess_to_onnx.py` script. This will read your raw `buffalo_detector.onnx` file from the `models/` directory and save a new, processed model named `scfrd_640_with_postprocessing2.onnx` back into the same directory.

```bash
python add_postprocess_to_onnx.py
```

You should see an output like this:

```
Pre-computing anchors...
Loading original model: models/buffalo_detector.onnx
Detached original outputs.
Added decoding nodes for all FPN levels.
...
Added NonMaxSuppression node with dynamic thresholds.
...
Graph cleanup complete.
Successfully saved new model to: models/scfrd_640_with_postprocessing_and_dynamic_thresholding.onnx
You can now run the inference script on this new model.
```

### Step 2: Run Detection and Alignment

Now you can use the new, processed model to detect faces in any image. The `buffalo_face_detection_and_alignment.py` script takes the image path as a required argument.

```bash
# Run on CPU
python buffalo_face_detection_and_alignment.py --image_path "path/to/your/test_image.jpg"

# Run on GPU (if available)
python buffalo_face_detection_and_alignment.py --image_path "path/to/your/test_image.jpg" --device cuda
```

### Step 3: Check the Results

After the script finishes, a new output directory (e.g., `output_test_image/`) will be created. Inside, you will find:

  * `test_image_original.jpg`: A copy of your input.
  * `face_0_cropped.jpg`: The cropped image of the first detected face.
  * `face_0_aligned_224x224.jpg`: The aligned image of the first face.
  * `face_1_cropped.jpg`: (and so on, for each detected face)
  * `face_1_aligned_224x224.jpg`: ...
  * `detection_results.json`: A JSON file containing all metadata.

**Example `detection_results.json`:**

```json
{
  "image_path": ".\\sample_image\\132708392331985247.png",
  "image_shape": [
    392,
    272,
    3
  ],
  "device": "cuda",
  "alignment_size": 224,
  "num_faces_detected": 1,
  "faces": [
    {
      "face_idx": 0,
      "bbox": [
        52.01586151123047,
        88.4870376586914,
        226.49713134765625,
        314.0374450683594
      ],
      "landmarks": [
        [
          80.88432312011719,
          179.15338134765625
        ],
        [
          154.16847229003906,
          177.8902587890625
        ],
        [
          103.39361572265625,
          222.6280059814453
        ],
        [
          91.40494537353516,
          263.6992492675781
        ],
        [
          146.775634765625,
          263.1143798828125
        ]
      ],
      "det_score": 0.8338995575904846,
      "detection_output": "output_132708392331985247\\face_0_cropped.jpg",
      "aligned_path": "output_132708392331985247\\face_0_aligned_224x224.jpg"
    }
  ]
}
```
