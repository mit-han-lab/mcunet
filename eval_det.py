import os
import argparse
import numpy as np

import torch
import tensorflow as tf
from PIL import Image, ImageDraw
from mcunet.utils.det_helper import MergeNMS, Yolo3Output

from mcunet.model_zoo import download_tflite

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # use only cpu for tf-lite evaluation

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--net_id', type=str, help='net id of the model')
# dataset args.
parser.add_argument('--image_path', default='assets/sample_images/person_det.jpg',
                    help='path to sample input image')

args = parser.parse_args()


def eval_image(image):
    interpreter.set_tensor(
        input_details[0]['index'], image.reshape(*input_shape))
    interpreter.invoke()
    output_data = [interpreter.get_tensor(
        output_details[i]['index']) for i in range(len(output_details))]
    # now parse the output in torch (the same logistics will be implemented on mcu side with tinyengine)
    outputs = [torch.from_numpy(d).permute(0, 3, 1, 2).contiguous() for d in output_data]
    outputs = [output_layer(output) for output_layer, output in zip(output_layers, outputs)]
    outputs = torch.cat(outputs, dim=1)
    ids, scores, bboxes = nms_layer(outputs)
    # now finally visualize the pred bboxes
    threshold = 0.3
    n_positive = (scores > threshold).sum()
    ids = ids[0, :n_positive, 0].numpy()  # single image
    bboxes = bboxes[0, :n_positive].numpy()
    pil_image = load_example_image(resolution[::-1])
    image_draw = ImageDraw.Draw(pil_image)
    for cls, bbox in zip(ids, bboxes):
        image_draw.rectangle(bbox, outline="red")
        print(cls, [round(_) for _ in bbox])
    filename, file_extension = os.path.splitext(args.image_path)
    vis_image_dir = filename + '_vis' + file_extension
    pil_image.save(vis_image_dir)


def load_example_image(resolution):
    image = Image.open(args.image_path).convert("RGB")
    image = image.resize(resolution)
    return image


def preprocess_image(image):
    image_np = np.array(image)[None, ...]
    image_np = (image_np / 255) * 2 - 1
    return image_np.astype('float32')  # since the graph has a quantizer input op, we use floating-point as input


def build_det_helper():
    nms = MergeNMS.build_from_config({
        "nms_name": "merge",
        "nms_valid_thres": 0.01,
        "nms_thres": 0.45,
        "nms_topk": 400,
        "post_nms": 100,
        "pad_val": -1,
    })
    output_configs = [
        {"num_class": 1, "anchors": [116, 90, 156, 198, 373, 326], "stride": 32, "alloc_size": [128, 128]},
        {"num_class": 1, "anchors": [30, 61, 62, 45, 59, 119], "stride": 16, "alloc_size": None},
        {"num_class": 1, "anchors": [10, 13, 16, 30, 33, 23], "stride": 8, "alloc_size": None},
    ]
    outputs = [
        Yolo3Output(**cfg).eval() for cfg in output_configs
    ]
    return nms, outputs


if __name__ == '__main__':
    tflite_path = download_tflite(net_id="person-det")
    interpreter = tf.lite.Interpreter(tflite_path)
    interpreter.allocate_tensors()

    nms_layer, output_layers = build_det_helper()

    # get input & output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    resolution = input_shape[1:3]  # we use non-square input for this model

    sample_image = load_example_image(resolution[::-1])  # w, h
    sample_image = preprocess_image(sample_image)

    eval_image(sample_image)
