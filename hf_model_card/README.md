---
license: apache-2.0
tags:
  - image-segmentation
  - unet
  - leaf
  - herbivory
  - computer-vision
library_name: pytorch
pipeline_tag: image-segmentation
---

# HerbivoR — leaf herbivory U-Net weights

Trained checkpoints for **[HerbivoR](https://github.com/mariosandovalmx/HerbivoR)** (or your local clone): contour / ROI completion and herbivory damage segmentation.

## Files

| File | Role |
|------|------|
| `best_unet_shape.pth` | Contour U-Net (mask-to-mask, 512 px) |
| `best_model.pth` | Damage U-Net (herbivory analysis) |

## Not included

- **MobileSAM** (`mobile_sam.pt`) — third-party weights (Apache-2.0). HerbivoR downloads them from [Ultralytics assets](https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt) via `download_models.py`. Original work: [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM).
- **BiRefNet_lite** — loaded from [`ZhengPeng7/BiRefNet_lite`](https://huggingface.co/ZhengPeng7/BiRefNet_lite) on first use.

## Usage

```bash
python download_models.py
# or
python download_models.py --repo mariosandovalmx/HerbivoR
```

## License

These HerbivoR-trained weights are released under **Apache License 2.0**. Please cite / credit HerbivoR if you use them in a publication or product.

## Citation

If you use these weights, please cite the HerbivoR project and the original papers for any third-party components you also use (MobileSAM, BiRefNet, Segment Anything).
