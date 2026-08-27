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

# Herbivora — leaf herbivory U-Net weights

Trained checkpoints for **[Herbivora](https://github.com/mariosandovalmx/Herbivora)** (or your local clone): contour / ROI completion and herbivory damage segmentation.

## Files

| File | Role |
|------|------|
| `best_unet_shape.pth` | Contour U-Net default (mask-to-mask, 512 px; Auto mode fallback) |
| `best_unet_shape_smooth.pth` | Contour specialist — entire / smooth margin |
| `best_unet_shape_serrated.pth` | Contour specialist — serrated margin |
| `best_unet_shape_lobed.pth` | Contour specialist — lobed margin |
| `best_unet_shape_compound.pth` | Contour specialist — compound leaves |
| `best_model.pth` | Damage U-Net (herbivory analysis) |

## Not included

- **MobileSAM** (`mobile_sam.pt`) — third-party weights (Apache-2.0). Herbivora downloads them from [Ultralytics assets](https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt) via `download_models.py`. Original work: [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM).
- **BiRefNet_lite** — loaded from [`ZhengPeng7/BiRefNet_lite`](https://huggingface.co/ZhengPeng7/BiRefNet_lite) on first use.

## Usage

```bash
python download_models.py
# or
python download_models.py --repo mariosandovalmx/Herbivora
```

## License

These Herbivora-trained weights (`best_unet_shape*.pth`, `best_model.pth`) are released under the
**[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)**
(same terms as the Herbivora application):

- **Allowed:** noncommercial research, education, and similar noncommercial use.
- **Not allowed without prior written permission:** commercial use (selling products/services, commercial workflows, redistribution for commercial purposes).
- **Attribution required:** if you use these weights in a publication, thesis, or presentation, cite Herbivora (see [`CITATION.cff`](https://github.com/mariosandovalmx/Herbivora/blob/main/CITATION.cff) in the GitHub repository).

Full terms: [`LICENSE`](LICENSE) in this repository and in the [Herbivora GitHub project](https://github.com/mariosandovalmx/Herbivora/blob/main/LICENSE).

Third-party components (MobileSAM, BiRefNet) remain under their own licenses.

## Citation

If you use these weights, please cite the Herbivora project and the original papers for any third-party components you also use (MobileSAM, BiRefNet, Segment Anything).
