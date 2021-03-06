#include <assert.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <chrono>
#include <iostream>
#include <opencv2/core/utility.hpp>
#include <vector>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/shape.hpp"

using namespace cv;

static const size_t N_INPUT = 1;
static const size_t N_OUTPUT = 5;

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

struct YOLACT {
    YOLACT(const char* model) {
        input_image_.resize(width_ * height_ * channels_);

        auto memory_info =  // TODO:
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        // Ort::AllocatorInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensor_ = Ort::Value::CreateTensor<float>(
            memory_info, input_image_.data(), input_image_.size(),
            input_shape_.data(), input_shape_.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(
            memory_info, results_.data(), results_.size(), output_shape_.data(),
            output_shape_.size());

        // initialize session options if needed
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        // OrtTensorRTProviderOptions trt_options{};
        // trt_options.device_id = 0;
        // trt_options.trt_max_workspace_size = 2147483648;
        // trt_options.trt_max_partition_iterations = 1000;
        // trt_options.trt_min_subgraph_size = 1;
        // trt_options.trt_engine_cache_enable = 1;
        // trt_options.trt_engine_cache_path = "cache";
        // trt_options.trt_fp16_enable = 1;
        // session_options.AppendExecutionProvider_TensorRT(trt_options);

        // Sets graph optimization level
        // Available levels are
        // ORT_DISABLE_ALL -> To disable all optimizations
        // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant
        // node removals) ORT_ENABLE_EXTENDED -> To enable extended
        // optimizations (Includes level 1 + more complex optimizations like
        // node fusions) ORT_ENABLE_ALL -> To Enable All possible opitmizations
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = Ort::Session(env, model, session_options);
    }

    int Run() {
        const char* input_names[] = {"input.1"};
        // TODO: fix
        const char* output_names[] = {"792", "992", "794", "801", "596"};

        session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_,
                     N_INPUT, output_names, output_tensors_.data(), N_OUTPUT);

        for (int i = 0; i < 5; i++) {
            bool is = output_tensors_[i].IsTensor();
        }

        return 0;
    }

    void inputInfo() {
        Ort::AllocatorWithDefaultOptions allocator;

        // print number of model input nodes
        size_t num_input_nodes = session_.GetInputCount();
        std::vector<const char*> input_node_names(num_input_nodes);
        std::vector<int64_t>
            input_node_dims;  // simplify... this model has
                              // only 1 input node {1,3, 550, 550}. Otherwise
                              // need vector<vector<>>

        printf("Number of inputs = %zu\n", num_input_nodes);

        // iterate over all input nodes
        for (int i = 0; i < num_input_nodes; i++) {
            // print input node names
            char* input_name = session_.GetInputName(i, allocator);
            printf("Input %d : name=%s\n", i, input_name);
            input_node_names[i] = input_name;

            // print input node types
            Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            ONNXTensorElementDataType type = tensor_info.GetElementType();
            printf("Input %d : type=%d\n", i, type);

            // print input shapes/dims
            input_node_dims = tensor_info.GetShape();
            printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
            for (int j = 0; j < input_node_dims.size(); j++)
                printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
        }

        std::vector<int64_t> output_node_dims;

        size_t num_outputs = session_.GetOutputCount();
        // iterate over all input nodes
        for (int i = 0; i < num_outputs; i++) {
            // print input node names
            char* output_name = session_.GetOutputName(i, allocator);
            printf("Output %d : name=%s\n", i, output_name);

            // print input node types
            Ort::TypeInfo type_info = session_.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            ONNXTensorElementDataType type = tensor_info.GetElementType();
            printf("output %d : type=%d\n", i, type);

            // print output shapes/dims
            output_node_dims = tensor_info.GetShape();
            printf("output %d : num_dims=%zu\n", i, output_node_dims.size());
            for (int j = 0; j < output_node_dims.size(); j++)
                printf("output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
        }
    }

    static constexpr const int width_ = 550;
    static constexpr const int height_ = 550;
    static constexpr const int channels_ = 3;
    std::vector<float> input_image_{};
    std::array<float, 19248 * 4> results_{};

   private:
    Ort::Session session_{nullptr};

    Ort::Value input_tensor_{nullptr};
    std::array<int64_t, 4> input_shape_{1, channels_, width_, height_};

    std::array<Ort::Value, N_OUTPUT> output_tensors_ = {
        Ort::Value(nullptr), Ort::Value(nullptr), Ort::Value(nullptr),
        Ort::Value(nullptr), Ort::Value(nullptr)};

    Ort::Value output_tensor_{nullptr};
    std::array<int64_t, 3> output_shape_{1, 19248, 4};
};

int main(int argc, char* argv[]) {
    if (argc == 1) {
        printf(
            "Argument are required.\n1) model.onnx path\n2) test image "
            "path\n",
            argc);
        return 1;
    }

    const char* model = argv[1];
    const char* image = argv[2];
    auto img = imread(image);

    auto inf = YOLACT(model);

    // TODO: 5 outputs :))))
    inf.inputInfo();

    int length = 2;
    for (int i = length - 1; i >= 0; i--) {
        img = imread(image);

        std::chrono::steady_clock::time_point begin =
            std::chrono::steady_clock::now();
        cv::resize(img, img, cv::Size(550, 550), 0, 0, cv::INTER_NEAREST);

        img.convertTo(img, CV_32FC1);
        img.reshape(1, img.total() * img.channels());
        if (!img.isContinuous()) img = img.clone();

        inf.input_image_.assign(
            (float*)img.data, (float*)img.data + img.total() * img.channels());

        inf.Run();

        std::chrono::steady_clock::time_point end =
            std::chrono::steady_clock::now();

        std::cout << "Time difference = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         end - begin)
                         .count()
                  << "[ms]" << std::endl;
    }

    return 0;
}