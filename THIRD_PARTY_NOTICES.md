# Third-Party Notices — Herbivora

Herbivora bundles or downloads software and model weights that remain under their
**own licenses**. This notice is provided for transparency. It is **not** a
grant of additional rights beyond each component’s license.

Herbivora application code and Herbivora-trained weights (`best_unet_shape.pth`,
`best_model.pth`) are under the **PolyForm Noncommercial License 1.0.0** — see
the root `LICENSE` file. Commercial use of Herbivora or those weights requires
**prior written permission** from the copyright holder.

---

## MobileSAM

| | |
|---|---|
| **Component** | MobileSAM weights (`mobile_sam.pt`) |
| **Authors / project** | Chaoning Zhang et al. — [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM) |
| **Typical distribution used by Herbivora** | [Ultralytics assets](https://github.com/ultralytics/assets/releases) (`mobile_sam.pt`) |
| **License** | Apache License 2.0 |

Herbivora does **not** re-license MobileSAM. You must comply with Apache-2.0 for
that component. Cite the MobileSAM / Segment Anything papers when appropriate.

---

## BiRefNet / BiRefNet_lite

| | |
|---|---|
| **Component** | BiRefNet_lite (segmentation background model) |
| **Source** | [ZhengPeng7/BiRefNet_lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite) on Hugging Face |
| **License** | As published by the model authors on Hugging Face / project repository (verify on the model card before commercial use) |

Herbivora loads BiRefNet_lite on first use. Those terms are separate from the
Herbivora PolyForm Noncommercial license.

---

## Other runtime dependencies

The installed Python environment also includes open-source libraries (for
example PyTorch, NumPy, OpenCV, CustomTkinter, Hugging Face `transformers`,
and related packages). Each package is distributed under its own license
(typically OSI-approved licenses such as BSD, MIT, Apache-2.0, or similar).

You can inspect package licenses in your install under `.venv` (e.g. each
package’s `LICENSE` / `METADATA` files) or on the package’s project page.

---

## Contact

Commercial licensing for **Herbivora** (software and Herbivora-trained weights):
see the project repository — https://github.com/mariosandovalmx/Herbivora
