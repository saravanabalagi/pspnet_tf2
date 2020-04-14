# PSPNet Tensorflow 2

Keras Pyramid Scene Parsing Network ported to tensorflow 2 from keras/tf_1.13.

- Caffe implementation: [PSPNet](https://github.com/hszhao/PSPNet)
- Py35 Keras Tensorflow1.13 implementation: [PSPNet-Keras-tensorflow](https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow)

## Dependencies

- Tensorflow 2 (tensorflow / tensorflow-gpu / tensorflow-cpu)
- OpenCV (opencv-python / opencv-contrib-python)

## Pretrained weights

Pretrained weights can be downloaded here:

| H5 Weights | Architecture | Numpy Weights |
|------------|--------------|---------------|
| [pspnet50_ade20k.h5](https://www.dropbox.com/s/7eyuzmag8df41j4/pspnet50_ade20k.h5?dl=0) | [pspnet50_ade20k.json](https://www.dropbox.com/s/xy7gs4g2def5z89/pspnet50_ade20k.json?dl=0) | [pspnet50_ade20k.npy](https://www.dropbox.com/s/z8la9ugpdss8k8q/pspnet50_ade20k.npy?dl=0) |
| [pspnet101_cityscapes.h5](https://www.dropbox.com/s/oymx9ktu6zrv7vz/pspnet101_cityscapes.h5?dl=0) | [pspnet101_cityscapes.json](https://www.dropbox.com/s/pofkdnf59nbs5w0/pspnet101_cityscapes.json?dl=0) | [pspnet101_cityscapes.npy](https://www.dropbox.com/s/2tdl01ihse7p9sr/pspnet101_cityscapes.npy?dl=0) |
| [pspnet101_voc2012.h5](https://www.dropbox.com/s/lqkmukeuo78cbcs/pspnet101_voc2012.h5?dl=0) | [pspnet101_voc2012.json](https://www.dropbox.com/s/i9f2p3q1d4wohd3/pspnet101_voc2012.json?dl=0) | [pspnet101_voc2012.npy](https://www.dropbox.com/s/yp4im80m72r6h98/pspnet101_voc2012.npy?dl=0) |

Download weights in 
- `.h5` and `.json` format and place them at `weights/keras` or
- `.npy` and place them at `weights/npy`

Find example [notebook](save_and_load.ipynb) which demonstrates save and load.

## Usage:

```sh
# python pspnet.py -m <model> -i <input_image>  -o <output_path> [-other_arguments]
python pspnet.py -m pspnet101_cityscapes -i example_images/cityscapes.jpg -o example_results/cityscapes.jpg -s -ms -f
python pspnet.py -m pspnet101_voc2012 -i example_images/pascal_voc.jpg -o example_results/pascal_voc.jpg -s -ms -f
python pspnet.py -m pspnet50_ade20k -i example_images/ade20k.jpg -o example_results/ade20k.jpg -s -ms -f
```
List of arguments:
```sh
 -m --model        - which model to use: 'pspnet50_ade20k', 'pspnet101_cityscapes', 'pspnet101_voc2012'
    --id           - (int) GPU Device id. Default 0
 -s --sliding      - Use sliding window
 -f --flip         - Additional prediction of flipped image
 -ms --multi_scale - Predict on multiscale images
```

![new](https://img.shields.io/badge/-new-blue) Batch Predict on GPU, check source [here](https://github.com/saravanabalagi/pspnet_tf2/blob/master/pspnet.py#L49)

## Keras results:

| Input | Segmented | Blended | Probe |
|-------|-----------|---------|-------|
| ![Original](example_images/ade20k.jpg) | ![New](example_results/ade20k_seg.jpg) | ![New](example_results/ade20k_seg_blended.jpg) | ![New](example_results/ade20k_probs.jpg) |
| ![Original](example_images/cityscapes.jpg) | ![New](example_results/cityscapes_seg.jpg) | ![New](example_results/cityscapes_seg_blended.jpg) | ![New](example_results/cityscapes_probs.jpg) |
| ![Original](example_images/pascal_voc.jpg) | ![New](example_results/pascal_voc_seg.jpg) | ![New](example_results/pascal_voc_seg_blended.jpg) | ![New](example_results/pascal_voc_probs.jpg) |

## Implementation 

* The interpolation layer is implemented as custom layer "Interp"
* Forward step takes about ~1 sec on single image
* Memory usage can be optimized with:
```python
# before calling any of the tf functions
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    # if you want to restrict total memory you can try
    # tf.config.experimental.set_memory_growth(gpu, True)
```
