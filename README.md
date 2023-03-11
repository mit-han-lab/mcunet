# MCUNet: Tiny Deep Learning on IoT Devices 

This is the official implementation of the MCUNet series.

###  [website](http://mcunet.mit.edu/) | [paper](https://arxiv.org/abs/2007.10319) | [paper (v2)](https://arxiv.org/abs/2110.15352) | [demo video](https://www.youtube.com/watch?v=F4XKn0iDfxg)

![demo](assets/figures/mcunet_demo.gif)

## News

**If you are interested in getting updates, please sign up [here](https://forms.gle/UW1uUmnfk1k6UJPPA) to get notified!**

- **(2022/12)** We simplified the `net_id` of models (new version: `mcunet-in0`, `mcunet-vww1`, etc.) for an upcoming review paper (stay tuned!).
- **(2022/10)** Our new work [On-Device Training Under 256KB Memory](https://arxiv.org/abs/2206.15472) is highlighted on the [MIT homepage](http://web.mit.edu/spotlight/learning-edge/)!
- **(2022/09)** Our new work [On-Device Training Under 256KB Memory](https://arxiv.org/abs/2206.15472) is accepted to NeurIPS 2022! It enables tiny on-device training for IoT devices [[demo](https://www.youtube.com/watch?v=XaDCO8YtmBw)]. 
- **(2022/08)** We release the source code of **TinyEngine** in [this repo](https://github.com/mit-han-lab/tinyengine). Please take a look!
- **(2022/08)** Our new course on **TinyML and Efficient Deep Learning** will be released soon in September 2022: [efficientml.ai](https://efficientml.ai/).
- **(2022/07)** We also include the person detection model used in the video demo above. We will also include the deployment code in TinyEngine release. 
- **(2022/06)** We refactor the MCUNet repo as a standalone repo (previous repo: https://github.com/mit-han-lab/tinyml)
- **(2021/10)** **MCUNetV2** is accepted to NeurIPS 2021: https://arxiv.org/abs/2110.15352 !
- **(2020/10)** **MCUNet** is accepted to NeurIPS 2020 as **spotlight**: https://arxiv.org/abs/2007.10319 !
- Our projects are covered by: [MIT News](https://news.mit.edu/2020/iot-deep-learning-1113), [MIT News (v2)](https://news.mit.edu/2021/tiny-machine-learning-design-alleviates-bottleneck-memory-usage-iot-devices-1208), [WIRED](https://www.wired.com/story/ai-algorithms-slimming-fit-fridge/), [Morning Brew](https://www.morningbrew.com/emerging-tech/stories/2020/12/07/researchers-figured-fit-ai-ever-onto-internet-things-microchips), [Stacey on IoT](https://staceyoniot.com/researchers-take-a-3-pronged-approach-to-edge-ai/), [Analytics Insight](https://www.analyticsinsight.net/amalgamating-ml-and-iot-in-smart-home-devices/), [Techable](https://techable.jp/archives/142462), etc.


## Overview

Microcontrollers are low-cost, low-power hardware. They are widely deployed and have wide applications.

![teaser](assets/figures/applications.png)

But the tight memory budget (50,000x smaller than GPUs) makes deep learning deployment difficult.

![teaser](assets/figures/memory_size.png)

MCUNet is a **system-algorithm co-design** framework for tiny deep learning on microcontrollers. It consists of **TinyNAS** and **TinyEngine**. They are co-designed to fit the tight memory budgets.

With system-algorithm co-design, we can significantly improve the deep learning performance on the same tiny memory budget.

![teaser](assets/figures/overview.png)

Our **TinyEngine** inference engine could be a useful infrastructure for MCU-based AI applications. It significantly **improves the inference speed and reduces the memory usage** compared to existing libraries like [TF-Lite Micro](https://www.tensorflow.org/lite/microcontrollers), [CMSIS-NN](https://arxiv.org/abs/1801.06601), [MicroTVM](https://tvm.apache.org/2020/06/04/tinyml-how-tvm-is-taming-tiny), etc. It improves the inference speed by **1.5-3x**, and reduces the peak memory by **2.7-4.8x**.

![teaser](assets/figures/latency_mem.png)



## Model Zoo

### Usage

You can build the pre-trained PyTorch `fp32` model or the `int8` quantized model in TF-Lite format.

```python
from mcunet.model_zoo import net_id_list, build_model, download_tflite
print(net_id_list)  # the list of models in the model zoo

# pytorch fp32 model
model, image_size, description = build_model(net_id="mcunet-in3", pretrained=True)  # you can replace net_id with any other option from net_id_list

# download tflite file to tflite_path
tflite_path = download_tflite(net_id="mcunet-in3")
```


### Evaluate

To evaluate the accuracy of PyTorch `fp32` models, run:

```bash
python eval_torch.py --net_id mcunet-in2 --dataset {imagenet/vww} --data-dir PATH/TO/DATA/val
```

To evaluate the accuracy of TF-Lite `int8` models, run:

```bash
python eval_tflite.py --net_id mcunet-in2 --dataset {imagenet/vww} --data-dir PATH/TO/DATA/val
```

### Model List

- Note that all the **latency**, **SRAM**, and **Flash** usage are profiled with **TinyEngine** on STM32F746.
- Here we only provide the `int8` quantized modes. `int4` quantized models (as shown in the paper) can further push the accuracy-memory trade-off, but lacking a general format support.
- For accuracy (top1, top-5), we report the accuracy of `fp32`/`int8` models respectively

The **ImageNet** model list:

| net_id              | MACs   | #Params | SRAM  | Flash  | Res. | Top-1<br />(fp32/int8) | Top-5<br />(fp32/int8) |
| ------------------- | ------ | ------- | ----- | ------ | ---- | ---------------------- | ---------------------- |
| *# baseline models* |        |         |       |        |      |                        |                        |
| mbv2-w0.35          | 23.5M  | 0.75M   | 308kB | 862kB  | 144  | 49.7%/49.0%            | 74.6%/73.8%            |
| proxyless-w0.3      | 38.3M  | 0.75M   | 292kB | 892kB  | 176  | 57.0%/56.2%            | 80.2%/79.7%            |
| *# mcunet models*   |        |         |       |        |      |                        |                        |
| mcunet-in0          | 6.4M   | 0.75M   | 266kB | 889kB  | 48   | 41.5%/40.4%            | 66.3%/65.2%            |
| mcunet-in1          | 12.8M  | 0.64M   | 307kB | 992kB  | 96   | 51.5%/49.9%            | 75.5%/74.1%            |
| mcunet-in2          | 67.3M  | 0.73M   | 242kB | 878kB  | 160  | 60.9%/60.3%            | 83.3%/82.6%            |
| mcunet-in3          | 81.8M  | 0.74M   | 293kB | 897kB  | 176  | 62.2%/61.8%            | 84.5%/84.2%            |
| mcunet-in4          | 125.9M | 1.73M   | 456kB | 1876kB | 160  | 68.4%/68.0%            | 88.4%/88.1%            |

The **VWW** model list:

*Note that the VWW dataset might be hard to prepare. You can download our pre-built `minival` set from [here](https://www.dropbox.com/s/bc7qi89ezra9711/vww-minival.tar?dl=0), around 380MB.*

| net_id      | MACs  | #Params | SRAM  | Flash | Resolution | Top-1<br />(fp32/int8) |
| ----------- | ----- | ------- | ----- | ----- | ---------- | ---------------------- |
| mcunet-vww0 | 6.0M  | 0.37M   | 146kB | 617kB | 64         | 87.4%/87.3%            |
| mcunet-vww1 | 11.6M | 0.43M   | 162kB | 689kB | 80         | 88.9%/88.9%            |
| mcunet-vww2 | 55.8M | 0.64M   | 311kB | 897kB | 144        | 91.7%/91.8%            |

For TF-Lite `int8` models, we do not use quantization-aware training (QAT), so some results is slightly lower than paper numbers. 

### Detection Model

We also share the person detection model used in the [demo](https://www.youtube.com/watch?v=F4XKn0iDfxg). To visualize the model's prediction on a sample image, please run the following command:

```bash
python eval_det.py
```

It will visualize the prediction here: `assets/sample_images/person_det_vis.jpg`.

The model takes in a small input resolution of 128x160 to reduce memory usage. It does not achieve state-of-the-art performance due to the limited image and model size but should provide decent performance for tinyML applications (please check the demo for a video recording). We will also release the deployment code in the upcoming TinyEngine release. 

## Requirement

- Python 3.6+

- PyTorch 1.4.0+

- Tensorflow 1.15 (if you want to test TF-Lite models; CPU support only)

## Acknowledgement

We thank MIT-IBM Watson AI Lab, Intel, Amazon, SONY, Qualcomm, NSF for supporting this research.


## Citation
If you find the project helpful, please consider citing our paper:

```
@article{lin2020mcunet,
  title={Mcunet: Tiny deep learning on iot devices},
  author={Lin, Ji and Chen, Wei-Ming and Lin, Yujun and Gan, Chuang and Han, Song},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

@inproceedings{
  lin2021mcunetv2,
  title={MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning},
  author={Lin, Ji and Chen, Wei-Ming and Cai, Han and Gan, Chuang and Han, Song},
  booktitle={Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2021}
} 

@article{
  lin2022ondevice, 
  title = {On-Device Training Under 256KB Memory},
  author = {Lin, Ji and Zhu, Ligeng and Chen, Wei-Ming and Wang, Wei-Chen and Gan, Chuang and Han, Song}, 
  journal = {arXiv:2206.15472 [cs]},
  url = {https://arxiv.org/abs/2206.15472},
  year = {2022}
}
```


## Related Projects

[On-Device Training Under 256KB Memory](https://tinytraining.mit.edu) (NeurIPS'22)

[TinyTL: Reduce Memory, Not Parameters for Efficient On-Device Learning](https://arxiv.org/abs/2007.11622) (NeurIPS'20)

[Once for All: Train One Network and Specialize it for Efficient Deployment](https://arxiv.org/abs/1908.09791) (ICLR'20)

[ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/pdf/1812.00332.pdf) (ICLR'19)

[AutoML for Architecting Efficient and Specialized Neural Networks](https://ieeexplore.ieee.org/abstract/document/8897011) (IEEE Micro)

[AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/pdf/1802.03494.pdf) (ECCV'18)

[HAQ: Hardware-Aware Automated Quantization](https://arxiv.org/pdf/1811.08886.pdf)  (CVPR'19, oral)
