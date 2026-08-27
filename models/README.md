# Herbivora model weights
#
# Place the required checkpoints here (or run `python download_models.py`).
# These files are git-ignored.
#
# Files
# -----
# mobile_sam.pt          (~39 MB)  MobileSAM — from Ultralytics assets (Apache-2.0)
#                                  https://github.com/ultralytics/assets/releases
# best_unet_shape.pth    (~93 MB)  U-Net Shape — Contour / ROI (Herbivora-trained)
#                                  https://huggingface.co/mariosandovalmx/Herbivora
#                                  License: PolyForm Noncommercial 1.0.0 (research/education)
# best_unet_shape_smooth.pth       Contour specialist (entire / smooth margin) — Hub repo
# best_unet_shape_serrated.pth     Contour specialist (serrated margin)
# best_unet_shape_lobed.pth        Contour specialist (lobed)
# best_unet_shape_compound.pth     Contour specialist (compound)
# best_model.pth         (~93 MB)  Damage U-Net — Analysis (Herbivora-trained)
#                                  https://huggingface.co/mariosandovalmx/Herbivora
#                                  License: PolyForm Noncommercial 1.0.0 (research/education)
#
# BiRefNet_lite is downloaded automatically on first Segmentation run
# (ZhengPeng7/BiRefNet_lite via Hugging Face transformers).
#
# MobileSAM is third-party (Chaoning Zhang et al.; Apache-2.0) and is NOT
# re-hosted in the Herbivora Hub repository.
#
# Herbivora-trained weights: noncommercial research/education only; commercial use
# requires prior written permission; attribution required (see repo LICENSE / CITATION.cff).
