#!/usr/bin/env python
import os
import argparse
import numpy as np
from tensorflow.keras.models import model_from_json, load_model
import layers_builder as layers
from glob import glob
from utils import utils
from tensorflow.keras.utils import CustomObjectScope
import cv2
import math

# # Fix for macOS, uncomment it
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# These are the means for the ImageNet pretrained ResNet
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order


class PSPNet(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017"""

    def __init__(self, nb_classes, resnet_layers, input_shape, weights):
        self.input_shape = input_shape
        self.num_classes = nb_classes

        json_path = os.path.join("weights", "keras", weights + ".json")
        h5_path = os.path.join("weights", "keras", weights + ".h5")
        if 'pspnet' in weights:
            if os.path.isfile(json_path) and os.path.isfile(h5_path):
                print("Keras model & weights found, loading...")
                with CustomObjectScope({'Interp': layers.Interp}):
                    with open(json_path) as file_handle:
                        self.model = model_from_json(file_handle.read())
                self.model.load_weights(h5_path)
            else:
                print("No Keras model & weights found, import from npy weights.")
                self.model = layers.build_pspnet(nb_classes=nb_classes,
                                                 resnet_layers=resnet_layers,
                                                 input_shape=self.input_shape)
                self.set_npy_weights(weights)
        else:
            print('Load pre-trained weights')
            self.model = load_model(weights)

    def predict(self, img, flip_evaluation=False):
        """
        Predict segmentation for an image.
        Arguments:
            img: must be rows x cols x 3
        """

        if img.shape[0:2] != self.input_shape:
            print(
                "Input %s not fitting for network size %s, resizing. "
                "You may want to try sliding prediction for better results." % (
                    img.shape[0:2], self.input_shape))
            img = cv2.resize(img, self.input_shape)

        img = img - DATA_MEAN
        img = img[:, :, ::-1]  # RGB => BGR
        img = img.astype('float32')

        probs = self.feed_forward(img, flip_evaluation)

        return probs

    def predict_sliding(self, full_img, flip_evaluation):
        """
        Predict on tiles of exactly the network input shape.
        This way nothing gets squeezed.
        """
        tile_size = self.input_shape
        classes = self.num_classes
        overlap = 1 / 3

        stride = math.ceil(tile_size[0] * (1 - overlap))
        tile_rows = max(int(math.ceil((full_img.shape[0] - tile_size[0]) / stride) + 1),
                        1)  # strided convolution formula
        tile_cols = max(int(math.ceil((full_img.shape[1] - tile_size[1]) / stride) + 1), 1)
        print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
        full_probs = np.zeros((full_img.shape[0], full_img.shape[1], classes))
        count_predictions = np.zeros((full_img.shape[0], full_img.shape[1], classes))
        tile_counter = 0
        for row in range(tile_rows):
            for col in range(tile_cols):
                x1 = int(col * stride)
                y1 = int(row * stride)
                x2 = min(x1 + tile_size[1], full_img.shape[1])
                y2 = min(y1 + tile_size[0], full_img.shape[0])
                x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
                y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

                img = full_img[y1:y2, x1:x2]
                padded_img = self.pad_image(img, tile_size)
                # plt.imshow(padded_img)
                # plt.show()
                tile_counter += 1
                print("Predicting tile %i" % tile_counter)
                padded_prediction = self.predict(padded_img, flip_evaluation)
                prediction = padded_prediction[0:img.shape[0], 0:img.shape[1], :]
                count_predictions[y1:y2, x1:x2] += 1
                full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

        # average the predictions in the overlapping regions
        full_probs /= count_predictions
        # visualize normalization Weights
        # plt.imshow(np.mean(count_predictions, axis=2))
        # plt.show()
        return full_probs

    @staticmethod
    def pad_image(img, target_size):
        """Pad an image up to the target size."""
        rows_missing = target_size[0] - img.shape[0]
        cols_missing = target_size[1] - img.shape[1]
        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')
        return padded_img

    def predict_multi_scale(self, img, flip_evaluation, sliding_evaluation, scales):
        """Predict an image by looking at it with different scales."""

        full_probs = np.zeros((img.shape[0], img.shape[1], self.num_classes))
        h_ori, w_ori = img.shape[:2]

        print("Started prediction...")
        for scale in scales:
            print("Predicting image scaled by %f" % scale)
            scaled_img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            if sliding_evaluation:
                scaled_probs = self.predict_sliding(scaled_img, flip_evaluation)
            else:
                scaled_probs = self.predict(scaled_img, flip_evaluation)

            # scale probs up to full size
            # visualize_prediction(probs)
            probs = cv2.resize(scaled_probs, (w_ori, h_ori))
            full_probs += probs
        full_probs /= len(scales)
        print("Finished prediction...")

        return full_probs

    def feed_forward(self, data, flip_evaluation=False):
        assert data.shape == (self.input_shape[0], self.input_shape[1], 3)

        if flip_evaluation:
            input_with_flipped = np.array([data, np.flip(data, axis=1)])
            prediction_with_flipped = self.model.predict(input_with_flipped)
            prediction = (prediction_with_flipped[0] + np.fliplr(prediction_with_flipped[1])) / 2.0
        else:
            prediction = self.model.predict(np.expand_dims(data, 0))[0]
        return prediction

    def set_npy_weights(self, weights_path):
        npy_weights_path = os.path.join("weights", "npy", weights_path + ".npy")
        json_path = os.path.join("weights", "keras", weights_path + ".json")
        h5_path = os.path.join("weights", "keras", weights_path + ".h5")

        print("Importing weights from %s" % npy_weights_path)
        weights = np.load(npy_weights_path, encoding='bytes', allow_pickle=True).item()
        for layer in self.model.layers:
            print(layer.name)
            if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
                mean = weights[layer.name.encode()]['mean'.encode()].reshape(-1)
                variance = weights[layer.name.encode()]['variance'.encode()].reshape(-1)
                scale = weights[layer.name.encode()]['scale'.encode()].reshape(-1)
                offset = weights[layer.name.encode()]['offset'.encode()].reshape(-1)

                self.model.get_layer(layer.name).set_weights(
                    [scale, offset, mean, variance])

            elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
                try:
                    weight = weights[layer.name.encode()]['weights'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight])
                except Exception as err:
                    biases = weights[layer.name.encode()]['biases'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight, biases])
        print('Finished importing weights.')

        print("Writing keras model & weights")
        json_string = self.model.to_json()
        with open(json_path, 'w') as file_handle:
            file_handle.write(json_string)
        self.model.save_weights(h5_path)
        print("Finished writing Keras model & weights")


class PSPNet50(PSPNet):
    """Build a PSPNet based on a 50-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=50,
                        input_shape=input_shape, weights=weights)


class PSPNet101(PSPNet):
    """Build a PSPNet based on a 101-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=101,
                        input_shape=input_shape, weights=weights)


def get_pspnet(model_name):
    pspnet_layers, pretrained_model_name = model_name.split("_")
    nb_classes_config = {
        'ade20k': 150,
        'cityscapes': 19,
        'voc2012': 21
    }
    pspnet_layers_config = {
        'ade20k': 'pspnet50',
        'cityscapes': 'pspnet101',
        'voc2012': 'pspnet101'
    }
    input_shape_config = {
        'ade20k': (473, 473),
        'cityscapes': (713, 713),
        'voc2012': (473, 473)
    }
    pspnet_config = {
        'pspnet50': PSPNet50,
        'pspnet101': PSPNet101
    }
    nb_classes = nb_classes_config[pretrained_model_name]
    pspnet_layers = pspnet_layers_config[pretrained_model_name]
    input_shape = input_shape_config[pretrained_model_name]
    pspnet_cls = pspnet_config[pspnet_layers]
    return pspnet_cls(nb_classes=nb_classes, input_shape=input_shape,
                      weights=f'{pspnet_layers}_{pretrained_model_name}')


def main(args):
    # Handle input and output args
    images = glob(args.glob_path) if args.glob_path else [args.input_path, ]
    if args.glob_path:
        fn, ext = os.path.splitext(args.output_path)
        if ext:
            parser.error("output_path should be a folder for multiple file input")
        if not os.path.isdir(args.output_path):
            os.mkdir(args.output_path)

    # Predict
    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    print(args)
    pspnet = get_pspnet(args.model)

    evaluation_scales = [1.0]
    if args.multi_scale:
        evaluation_scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # must be all floats! Taken from original paper

    print(f"[{'ON' if args.multi_scale else 'OFF'}] Multi scale evaluation, scales: {evaluation_scales}")
    print(f"[{'ON' if args.flip else 'OFF'}] Flipped evaluation")

    for i, img_path in enumerate(images):
        print("Processing image {} / {}".format(i + 1, len(images)))
        img = cv2.imread(img_path)[:, :, ::-1]  # Read RGB

        probs = pspnet.predict_multi_scale(img, args.flip, args.sliding, evaluation_scales)

        cm = np.argmax(probs, axis=2)
        pm = np.max(probs, axis=2)

        colored_class_image = utils.color_class_image(cm, args.model)
        alpha_blended = 0.5 * colored_class_image + 0.5 * img

        if args.glob_path:
            input_filename, ext = os.path.splitext(os.path.basename(img_path))
            filename = os.path.join(args.output_path, input_filename)
        else:
            filename, ext = os.path.splitext(args.output_path)

        cv2.imwrite(filename + "_seg" + ext, colored_class_image)
        cv2.imwrite(filename + "_probs" + ext, (pm * 255).astype(np.uint8))
        cv2.imwrite(filename + "_seg_blended" + ext, alpha_blended)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet101_voc2012',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('-w', '--weights', type=str, default=None)
    parser.add_argument('-i', '--input_path', type=str, default='example_images/ade20k.jpg',
                        help='Path the input image')
    parser.add_argument('-g', '--glob_path', type=str, default=None,
                        help='Glob path for multiple images')
    parser.add_argument('-o', '--output_path', type=str, default='example_results/ade20k.jpg',
                        help='Path to output')
    parser.add_argument('--id', default="0")
    parser.add_argument('--input_size', type=int, default=500)
    parser.add_argument('-s', '--sliding', action='store_true',
                        help="Whether the network should be slided over the original image for prediction.")
    parser.add_argument('-f', '--flip', action='store_true', default=True,
                        help="Whether the network should predict on both image and flipped image.")
    parser.add_argument('-ms', '--multi_scale', action='store_true',
                        help="Whether the network should predict on multiple scales.")

    args = parser.parse_args()

    main(args)
