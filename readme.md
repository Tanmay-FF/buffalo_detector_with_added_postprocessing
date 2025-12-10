
# ONNX Face Detection and Alignment Pipeline

This project provides a complete pipeline for high-performance face detection and alignment using ONNX models. It features two main components:

1. **Python-based pipeline** that embeds post-processing (anchor decoding and NMS) into the ONNX model and performs detection + alignment.
2. **C++-based inference environment** (via Docker) for high-performance deployment using ONNX Runtime GPU and CUDA 11.8.

---

## 🚀 Features

- **High-Performance Inference:** Supports both Python and C++ inference using `onnxruntime` (CPU or GPU).
- **Embedded Post-processing:** The `add_postprocess_to_onnx.py` script adds decoding and NMS directly into the ONNX graph — simplifying downstream deployment.
- **Face Alignment:** Performs 5-point landmark-based alignment to generate standardized 224×224 ArcFace-aligned faces.
- **C++ Docker Environment:** Includes a `Dockerfile` that sets up ONNX Runtime GPU (CUDA 11.8) and OpenCV for C++ inference.
- **Comprehensive Output:** Each input image produces:
  - Cropped and aligned face images.
  - A detailed `detection_results.json` metadata file.
- **Dynamic Thresholds:** The enhanced model supports runtime control of score and IoU thresholds.

---

## 📦 Project Structure

```
buffalo_detector_with_added_postprocessing/
├── models/
│   ├── buffalo_detector.onnx                                       # (INPUT) Raw model
│   └── scfrd_640_with_postprocessing_and_dynamic_thresholding.onnx # (OUTPUT) Model with NMS + decoding
│
├── add_postprocess_to_onnx.py                  # Adds post-processing to model
├── buffalo_face_detection_and_alignment.py     # Python inference script
├── main.cpp                                    # C++ inference and alignment code
├── Dockerfile                                  # Docker setup for ONNX Runtime C++ (CUDA 11.8)
├── requirements.txt                            # Python dependencies
└── README.md
```

---

## 🧰 Setup and Installation

### Option 1️⃣ — Python Environment

1. **Clone and Setup:**
   ```bash
   git clone https://github.com/Tanmay-FF/buffalo_detector_with_added_postprocessing.git
   cd buffalo_detector_with_added_postprocessing
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Prepare Models:**
   Place your raw model in:
   ```
   models/buffalo_detector.onnx
   ```

3. **Add Post-Processing to Model:**
   ```bash
   python add_postprocess_to_onnx.py
   ```
   This creates:
   ```
   models/scfrd_640_with_postprocessing_and_dynamic_thresholding.onnx
   ```

4. **Run Detection and Alignment (Python):**
   ```bash
   python buffalo_face_detection_and_alignment.py --image_path "path/to/test.jpg" --device cuda
   ```

---

### Option 2️⃣ — C++ Docker Environment (CUDA 11.8)

You can now build and run the **C++ ONNX Runtime GPU environment** using the provided `Dockerfile`.  
This setup includes:
- CUDA 11.8 + cuDNN 8  
- ONNX Runtime GPU (v1.16.3)  
- OpenCV (for image I/O and alignment)

#### 🏗️ Build the Docker Image

```bash
docker build -t buffalo_cpp_env .
```

#### ▶️ Run the Container

Mount your local `models` and `sample_image` directories inside the container:
```bash
docker run --gpus all -it \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/sample_image:/workspace/sample_image \
  buffalo_cpp_env
```

#### 🧪 Run the C++ Inference Program

Inside the container:
```bash
 ./main models/scfrd_640_with_postprocessing_and_dynamic_thresholding.onnx  sample_image/0_Parade_Parade_0_178.jpg "cuda"
```

This program:
- Loads the ONNX model (`models/scfrd_640_with_postprocessing_and_dynamic_thresholding.onnx`)
- Runs detection on the input image
- Performs alignment using 5-point landmarks
- Saves:
  - Cropped face images
  - 224×224 aligned faces

✅ **Results from C++ match the Python inference outputs**, ensuring consistent detection and alignment.

---

## 🧾 Example Output

Output directory (e.g., `output_test_image/`):

```
output_test_image/
├── test_image_original.jpg
├── face_0_cropped.jpg
├── face_0_aligned_224x224.jpg
├── face_1_cropped.jpg
├── face_1_aligned_224x224.jpg
└── detection_results.json
```

---

## 🧩 Example `detection_results.json`

```json
{
  "image_path": "./sample_image/132708392331985247.png",
  "device": "cuda",
  "alignment_size": 224,
  "num_faces_detected": 1,
  "faces": [
    {
      "face_idx": 0,
      "bbox": [52.01586, 88.48704, 226.4971, 314.0374],
      "landmarks": [
        [80.88432, 179.15338],
        [154.16847, 177.89026],
        [103.39361, 222.62800],
        [91.40494, 263.69925],
        [146.77563, 263.11438]
      ],
      "det_score": 0.8339,
      "detection_output": "output_132708392331985247/face_0_cropped.jpg",
      "aligned_path": "output_132708392331985247/face_0_aligned_224x224.jpg"
    }
  ]
}
```

---


**Author:** Tanmay Thaker
**Docker GPU Runtime:** CUDA 11.8  
**ONNX Runtime:** v1.16.3  
**OpenCV:** Latest (via apt)
````
