---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://polyformproject.org/licenses/noncommercial/1.0.0/
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

These HerbivoR-trained weights (`best_unet_shape.pth`, `best_model.pth`) are released under the
**[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)**
(same terms as the HerbivoR application):

- **Allowed:** noncommercial research, education, and similar noncommercial use.
- **Not allowed without prior written permission:** commercial use (selling products/services, commercial workflows, redistribution for commercial purposes).
- **Attribution required:** if you use these weights in a publication, thesis, or presentation, cite HerbivoR (see [`CITATION.cff`](https://github.com/mariosandovalmx/HerbivoR/blob/main/CITATION.cff) in the GitHub repository).

Full terms: [`LICENSE`](LICENSE) in this repository and in the [HerbivoR GitHub project](https://github.com/mariosandovalmx/HerbivoR/blob/main/LICENSE).

Third-party components (MobileSAM, BiRefNet) remain under their own licenses.

## Citation

If you use these weights, please cite the HerbivoR project and the original papers for any third-party components you also use (MobileSAM, BiRefNet, Segment Anything).
