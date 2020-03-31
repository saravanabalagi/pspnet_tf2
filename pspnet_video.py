#!/usr/bin/env python
"""
This module is a Keras/Tensorflow based implementation of Pyramid Scene Parsing Networks.
Original paper & code published by Hengshuang Zhao et al. (2017)
"""
import argparse
import numpy as np
from pspnet import get_pspnet
from utils import utils
import cv2
import datetime
from tensorflow.python.client import device_lib
from os.path import splitext

# These are the means for the ImageNet pretrained ResNet
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order
EVALUATION_SCALES = [1.0]  # must be all floats!


def get_gpu_name():
    local_device_protos = device_lib.list_local_devices()
    l = [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']
    s = ''
    for t in l:
        s += t[t.find("name: ") + len("name: "):t.find(", pci")] + " "
    return s


GPU_NAME = get_gpu_name()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet50_ade20k',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('-i', '--input_path', type=str, default='example_images/ade20k.jpg',
                        help='Path the input image')
    parser.add_argument('-o', '--output_path', type=str, default='example_results/ade20k.jpg',
                        help='Path to output')
    parser.add_argument('--id', default="1")
    parser.add_argument('-s', '--sliding', action='store_true',
                        help="Whether the network should be slided over the original image for prediction.")
    parser.add_argument('-f', '--flip', action='store_true',
                        help="Whether the network should predict on both image and flipped image.")
    parser.add_argument('-ms', '--multi_scale', action='store_true',
                        help="Whether the network should predict on multiple scales.")
    args = parser.parse_args()

    # environ["CUDA_VISIBLE_DEVICES"] = args.id

    cap = cv2.VideoCapture(args.input_path)
    print(args)
    counter = 0

    pspnet = get_pspnet(args.model)
    if args.multi_scale:
        # EVALUATION_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # must be all floats!
        EVALUATION_SCALES = [0.15, 0.25, 0.5]  # must be all floats!

    time_sum = 0
    while True:
        # Capture frame-by-frame
        ret, img = cap.read()
        if img is None:
            break

        # img = cv2.resize(img,(int(16.0*713/9.0),713))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start = datetime.datetime.now()
        if args.multi_scale:
            probs = pspnet.predict_multi_scale(img, args.flip, args.sliding, EVALUATION_SCALES)
        else:
            probs = pspnet.predict(img, args.flip)

        # End time, Time elapsed
        end = datetime.datetime.now()
        diff = end - start

        cm = np.argmax(probs, axis=2)
        pm = np.max(probs, axis=2)

        colored_class_image = utils.color_class_image(cm, args.model)
        alpha_blended = 0.5 * colored_class_image + 0.5 * img

        time_sum += diff.microseconds / 1000.0
        print(counter, diff.microseconds / 1000.0, 'ms')

        cv2.putText(alpha_blended, '%s %s' % (GPU_NAME, args.model), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0),
                    16, cv2.LINE_AA)
        cv2.putText(alpha_blended, '%s %s' % (GPU_NAME, args.model), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (255, 255, 255), 10, cv2.LINE_AA)

        cv2.putText(alpha_blended, 'Prediction time: %.0fms (%.1f fps) AVG: %.0fms (%.1f fps)' % (
            diff.microseconds / 1000.0, 1000000.0 / diff.microseconds, time_sum / (counter + 1),
            1000.0 / (time_sum / (counter + 1))), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 16, cv2.LINE_AA)
        cv2.putText(alpha_blended, 'Prediction time: %.0fms (%.1f fps) AVG: %.0fms (%.1f fps)' % (
            diff.microseconds / 1000.0, 1000000.0 / diff.microseconds, time_sum / (counter + 1),
            1000.0 / (time_sum / (counter + 1))), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10,
                    cv2.LINE_AA)

        filename, ext = splitext(args.output_path)
        cv2.imwrite(filename + "_%08d_seg" % counter + ext, colored_class_image)
        cv2.imwrite(filename + "_%08d_probs" % counter + ext, (pm * 255).astype(np.uint8))
        cv2.imwrite(filename + "_%08d_seg_blended" % counter + ext, alpha_blended)
        counter = counter + 1
