# HerbivoR model weights
#
# Place the three required checkpoints here (or run `python download_models.py`).
# These files are git-ignored; download them from Hugging Face:
#   https://huggingface.co/mariosandovalmx/HerbivoR
#
# Files
# -----
# mobile_sam.pt          (~39 MB)  MobileSAM — Segmentation (BiRefNet + MobileSAM)
# best_unet_shape.pth    (~93 MB)  U-Net Shape — Contour / ROI (mask-to-mask, 512 px)
# best_model.pth         (~93 MB)  Damage U-Net — Analysis (herbivory damage)
#
# BiRefNet_lite is downloaded automatically on first Segmentation run
# (ZhengPeng7/BiRefNet_lite via Hugging Face transformers).
