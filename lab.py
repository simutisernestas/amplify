from utils.augmentations import FastBaseTransform
import cv2
import numpy as np
import torch
import onnxruntime as rt
import time
import layers
import eval
import matplotlib.pyplot as plt
import time 
eval.get_args()
import glob
from pathlib import Path
Path("results").mkdir(exist_ok=True)
import imutils

sess_options = rt.SessionOptions()
# sess_options.enable_profiling = False
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

print("Loading trt execution, this may take a while... ")
sess = rt.InferenceSession("yolact.onnx", sess_options)
# sess.set_providers(['CUDAExecutionProvider'])
# sess.set_providers(['TensorrtExecutionProvider'])
if "TensorrtExecutionProvider" not in sess.get_providers():
    raise Exception("TensorrtExecutionProvider can't be executed with TensorRT")
input_name = sess.get_inputs()[0].name
loc_name = sess.get_outputs()[0].name
conf_name = sess.get_outputs()[1].name
mask_name = sess.get_outputs()[2].name
priors_name = sess.get_outputs()[3].name
proto_name = sess.get_outputs()[4].name

def inference(img_path, show=False):
    # if too big to fit on GPU
    
    for i in range(100):
        img = cv2.imread(img_path)

        start = time.time()

        img = imutils.resize(img, width=550, height=550)
        frame = torch.Tensor(img).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        
        # inference benchmark
        pred_onx = sess.run([loc_name, conf_name, mask_name, priors_name, proto_name], {input_name: batch.cpu().detach().numpy()})
        print(f"Inference duration: {(time.time()-start)}")
        
        detection = layers.Detect(81, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)
        preds = detection({'loc': torch.from_numpy(pred_onx[0]), 
                        'conf': torch.from_numpy(pred_onx[1]), 
                        'mask': torch.from_numpy(pred_onx[2]), 
                        'priors': torch.from_numpy(pred_onx[3]),
                        'proto': torch.from_numpy(pred_onx[4])})
        

    img = eval.prep_display(preds, frame.cpu(), None, None, None, None, undo_transform=False)
    cv2.imwrite('results/' + img_path.split('/')[1], img)

    if show:
        plt.imshow(img)
        plt.show()

print("Starting inference...")
for f in glob.glob("data/*.jpg"):
    print(f"Processing {f}")
    inference(f)
