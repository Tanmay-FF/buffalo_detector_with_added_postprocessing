#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

const int INPUT_W = 640;
const int INPUT_H = 640;
const float CONF_THRESH = 0.4f;
const float NMS_THRESH  = 0.4f;
const int ALIGN_SIZE = 224;

// Standard ArcFace destination landmarks
cv::Mat getArcFaceDst(int image_size = 224) {
    cv::Mat dst(5, 2, CV_32F);
    dst.at<float>(0, 0) = 38.2946f; dst.at<float>(0, 1) = 51.6963f;
    dst.at<float>(1, 0) = 73.5318f; dst.at<float>(1, 1) = 51.5014f;
    dst.at<float>(2, 0) = 56.0252f; dst.at<float>(2, 1) = 71.7366f;
    dst.at<float>(3, 0) = 41.5493f; dst.at<float>(3, 1) = 92.3655f;
    dst.at<float>(4, 0) = 70.7299f; dst.at<float>(4, 1) = 92.2041f;

    float ratio = image_size / 112.0f;
    dst *= ratio;
    return dst;
}

cv::Mat preprocess(const cv::Mat& img, std::vector<float>& blob, float& scale)
{
    int h = img.rows, w = img.cols;
    float im_ratio = (float)h / w;
    float model_ratio = (float)INPUT_H / INPUT_W;
    int new_w, new_h;

    if (im_ratio > model_ratio) {
        new_h = INPUT_H;
        new_w = int(new_h / im_ratio);
    } else {
        new_w = INPUT_W;
        new_h = int(new_w * im_ratio);
    }

    scale = float(new_h) / h;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    cv::Mat canvas(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(canvas(cv::Rect(0, 0, resized.cols, resized.rows)));

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F);
    rgb = (rgb - 127.5f) / 128.0f;

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    blob.clear();
    for (int c = 0; c < 3; ++c)
        blob.insert(blob.end(), (float*)channels[c].datastart, (float*)channels[c].dataend);

    return rgb;
}

cv::Mat alignFace(const cv::Mat& img, const cv::Mat& landmarks, int output_size = ALIGN_SIZE)
{
    cv::Mat dst_pts = getArcFaceDst(output_size);
    cv::Mat M = cv::estimateAffinePartial2D(landmarks, dst_pts);
    cv::Mat aligned;
    cv::warpAffine(img, aligned, M, cv::Size(output_size, output_size), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return aligned;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        cerr << "Usage: ./scrfd_infer <model_path.onnx> <image_path> [device]\n";
        return 1;
    }

    string model_path = argv[1];
    string image_path = argv[2];
    string device = (argc > 3) ? argv[3] : "cpu";

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        cerr << "Could not read image: " << image_path << endl;
        return 1;
    }

    cout << "Loaded image: " << image_path << " (" << img.cols << "x" << img.rows << ")\n";

    // --- Initialize ONNX Runtime ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "scrfd_infer");
    Ort::SessionOptions session_options;
    OrtCUDAProviderOptions cuda_options;

    if (device == "cuda") {
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        cout << "Using CUDA Execution Provider\n";
    } else {
        cout << "Using CPU Execution Provider\n";
    }

    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input/output names
    std::vector<const char*> input_names, output_names;
    size_t num_inputs = session.GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i)
        input_names.push_back(session.GetInputNameAllocated(i, allocator).release());

    size_t num_outputs = session.GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i)
        output_names.push_back(session.GetOutputNameAllocated(i, allocator).release());

    // --- Preprocess image ---
    std::vector<float> blob;
    float scale;
    preprocess(img, blob, scale);

    std::array<int64_t, 4> input_shape{1, 3, INPUT_H, INPUT_W};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, blob.data(), blob.size(), input_shape.data(), input_shape.size());

    // --- Prepare thresholds ---
    float score_thresh_val = CONF_THRESH;
    float iou_thresh_val = NMS_THRESH;
    std::array<int64_t, 1> thresh_shape{1};
    Ort::Value score_thresh_tensor = Ort::Value::CreateTensor<float>(mem_info, &score_thresh_val, 1, thresh_shape.data(), thresh_shape.size());
    Ort::Value iou_thresh_tensor = Ort::Value::CreateTensor<float>(mem_info, &iou_thresh_val, 1, thresh_shape.data(), thresh_shape.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    input_tensors.push_back(std::move(iou_thresh_tensor));
    input_tensors.push_back(std::move(score_thresh_tensor));

    // --- Run inference ---
    cout << "Running inference..." << endl;
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
                                      input_tensors.size(), output_names.data(), output_names.size());
    cout << "Inference complete." << endl;

    if (output_tensors.size() < 3) {
        cerr << "Unexpected number of outputs.\n";
        return 1;
    }

    const float* boxes = output_tensors[0].GetTensorData<float>();
    const float* scores = output_tensors[1].GetTensorData<float>();
    const float* kps = output_tensors[2].GetTensorData<float>();
    auto boxes_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t num_dets = boxes_shape[0];
    cout << "Detections: " << num_dets << endl;

    fs::path img_path(image_path);
    string base_name = img_path.stem().string();
    fs::create_directory("output_" + base_name);
    string out_dir = "output_" + base_name;

    for (size_t i = 0; i < num_dets; ++i) {
        float x1 = boxes[i * 4 + 0] / scale;
        float y1 = boxes[i * 4 + 1] / scale;
        float x2 = boxes[i * 4 + 2] / scale;
        float y2 = boxes[i * 4 + 3] / scale;
        float score = scores[i];

        cout << "Face " << i + 1 << " | Score: " << score
             << " | BBox: [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]\n";

        if (score < CONF_THRESH) continue;

        cv::Rect roi{cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2))};
        roi = roi & cv::Rect(0, 0, img.cols, img.rows);
        cv::Mat cropped = img(roi);
        string crop_path = out_dir + "/face_" + to_string(i) + "_crop.jpg";
        cv::imwrite(crop_path, cropped);

        // --- Landmark reshape [5x2]
        cv::Mat landmarks(5, 2, CV_32F);
        for (int j = 0; j < 5; ++j) {
            landmarks.at<float>(j, 0) = kps[i * 10 + j * 2 + 0] / scale;
            landmarks.at<float>(j, 1) = kps[i * 10 + j * 2 + 1] / scale;
        }

        // --- Alignment ---
        cv::Mat aligned = alignFace(img, landmarks, ALIGN_SIZE);
        string align_path = out_dir + "/face_" + to_string(i) + "_aligned_224x224.jpg";
        cv::imwrite(align_path, aligned);
    }

    cout << "All detections processed and saved in: " << out_dir << endl;
    return 0;
}
