  // //*************************************************************************
  // // initialize  enviroment...one enviroment per process
  // // enviroment maintains thread pools and other state info
  // Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  // // initialize session options if needed
  // Ort::SessionOptions session_options;
  // session_options.SetIntraOpNumThreads(1);

  // OrtTensorRTProviderOptions trt_options{};
  // trt_options.device_id = 0;
  // trt_options.trt_max_workspace_size = 2147483648;
  // trt_options.trt_max_partition_iterations = 10;
  // trt_options.trt_min_subgraph_size = 5;
  // trt_options.trt_engine_cache_enable = 1;
  // trt_options.trt_engine_cache_path = "cache";
  // trt_options.trt_dump_subgraphs = 1;
  // session_options.AppendExecutionProvider_TensorRT(trt_options);

  // // trt_options.trt_fp16_enable = 1;
  // // trt_options.trt_int8_enable = 1;
  // // trt_options.trt_int8_use_native_calibration_table = 1;

  // // Sets graph optimization level
  // // Available levels are
  // // ORT_DISABLE_ALL -> To disable all optimizations
  // // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant
  // node
  // // removals)
  // // ORT_ENABLE_EXTENDED -> To enable extended optimizations
  // // (Includes level 1 + more complex optimizations like node fusions)
  // // ORT_ENABLE_ALL -> To Enable All possible opitmizations
  // session_options.SetGraphOptimizationLevel(
  //     GraphOptimizationLevel::ORT_ENABLE_ALL);

  // const char* model_path = "../yolact.onnx";

  // printf("Using Onnxruntime C++ API\n");
  // Ort::Session session(env, model_path, session_options);

  // //*************************************************************************
  // // print model input layer (node names, types, shape etc.)
  // Ort::AllocatorWithDefaultOptions allocator;

  // size_t input_tensor_size =
  //     550 * 550 *
  //     3;  // simplify ... using known dim values to calculate size
  //         // use OrtGetTensorShapeElementCount() to get official size!

  // std::vector<int64_t>
  //     output_node_dims;  // simplify... this model has only 1 input node
  //     {1,
  //                        // 3, 550, 550}. Otherwise need vector<vector<>>

  // std::vector<float> input_tensor_values(input_tensor_size);
  // std::vector<const char*> output_node_names = {"792"};

  // for (int k = 0; k < 100; ++k) {
  //     auto img = imread("../data/stefan.jpg");

  //     std::chrono::steady_clock::time_point begin =
  //         std::chrono::steady_clock::now();
  //     cv::resize(img, img, cv::Size(550, 550));

  //     img.convertTo(img, CV_32FC1);
  //     img.reshape(1, img.total() * img.channels());
  //     if (!img.isContinuous()) img = img.clone();

  //     // int channels = img.channels();
  //     // int nRows = img.rows;
  //     // int nCols = img.cols * channels;

  //     // cv::Mat flat = img.reshape(1, img.total() * img.channels());
  //     // input_tensor_values = img.isContinuous() ? flat : flat.clone();

  //     // if (img.isContinuous()) {
  //     //     nCols *= nRows;
  //     //     nRows = 1;
  //     // }

  //     // auto size = img.size();
  //     // accept only char type matrices
  //     // CV_Assert(img.depth() == CV_8U);

  //     // int i, j;
  //     // uchar* p;
  //     // // uint count{0};
  //     // for (i = 0; i < nRows; ++i) {
  //     //     p = img.ptr<uchar>(i);
  //     //     for (j = 0; j < nCols; ++j) {
  //     //         // printf("%d %d %d %d", p[j], channels, nRows, nCols);
  //     //         // printf("%d ", p[j]);
  //     //         // count++;
  //     //         input_tensor_values[(i * nRows) + j] = (float)p[j];
  //     //         // input_tensor_values.push_back((float)p[j]);
  //     //     }
  //     // }
  //     // printf("%d ", count);

  //     // initialize input data with values in [0.0, 1.0]
  //     // for (unsigned int i = 0; i < input_tensor_size; i++)
  //     //     input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  //     // create input tensor object from data values
  //     auto memory_info =
  //         Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  //     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
  //         memory_info, (float*)img.data, input_tensor_size,
  //         input_node_dims.data(), 4);
  //     assert(input_tensor.IsTensor());

  //     // score model & input tensor, get back output tensor

  //     auto output_tensors =
  //         session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
  //                     &input_tensor, 1, output_node_names.data(), 1);
  //     assert(output_tensors.size() == 1 &&
  //     output_tensors.front().IsTensor());

  //     std::chrono::steady_clock::time_point end =
  //         std::chrono::steady_clock::now();

  //     std::cout << "Time difference = "
  //               << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                      end - begin)
  //                      .count()
  //               << "[ms]" << std::endl;

  //     cv::resize(img, img, cv::Size(7360, 4912));  //(7360, 4912, 3)
  // }

  // // Get pointer to output tensor float values
  // // float* floatarr =
  // output_tensors.front().GetTensorMutableData<float>();
  // // assert(abs(floatarr[0] - 0.000045) < 1e-6);