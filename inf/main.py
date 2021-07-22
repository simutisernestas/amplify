import onnxruntime as rt
import cv2
import numpy as np
import time
import glob

sess_options = rt.SessionOptions()
# sess_options.enable_profiling = False
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

print("Loading trt execution, this may take a while... ")
sess = rt.InferenceSession("yolact.onnx", sess_options)
# sess.set_providers(['CUDAExecutionProvider'])
# sess.set_providers(['TensorrtExecutionProvider'])
if "TensorrtExecutionProvider" not in sess.get_providers():
    raise Exception(
        "TensorrtExecutionProvider can't be executed with TensorRT")

input_name = sess.get_inputs()[0].name
loc_name = sess.get_outputs()[0].name
conf_name = sess.get_outputs()[1].name
mask_name = sess.get_outputs()[2].name
priors_name = sess.get_outputs()[3].name
proto_name = sess.get_outputs()[4].name


def inference(img_path):
    for i in range(5):
        img = cv2.imread(img_path)

        start = time.time()

        img = cv2.resize(img, (550, 550), interpolation=cv2.INTER_NEAREST)
        # Convert to BGR
        img = np.array(img)[:, :, [2, 1, 0]].astype('float32')
        # HWC -> CHW
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)

        # inference benchmark
        # pred_onx = sess.run([loc_name, conf_name, mask_name, priors_name, proto_name], {input_name: batch.cpu().detach().numpy()})
        pred_onx = sess.run([loc_name, conf_name, mask_name,
                             priors_name, proto_name], {input_name: img})
        print(f"Inference duration: {(time.time()-start)}")

        preds = {'loc': pred_onx[0],
                 'conf': pred_onx[1],
                 'mask': pred_onx[2],
                 'priors': pred_onx[3],
                 'proto': pred_onx[4]}


print("Starting inference...")
for f in glob.glob("data/*.jpg"):
    print(f"Processing {f}")
    inference(f)
