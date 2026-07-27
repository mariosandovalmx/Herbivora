"""
U-Net model using Segmentation Models PyTorch (SMP) with a pre-trained encoder.

Why SMP + pretrained encoder:
  - Training from scratch with ~5 images produces heavy noise and fragmentation.
  - A ResNet/EfficientNet backbone pre-trained on ImageNet already "knows" edges,
    textures, and shapes, so the decoder only needs to learn the segmentation head.

Output: [B, num_classes, H, W] logits (no Softmax — CrossEntropyLoss handles it).
"""

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError(
        "Install segmentation_models_pytorch:\n"
        "  pip install segmentation-models-pytorch"
    )


def build_model(encoder_name="resnet50", num_classes=4, pretrained=True):
    """
    Build a U-Net with a pre-trained encoder.

    Args:
        encoder_name: Backbone name. Good choices:
                        - 'resnet50'        (solid, widely tested)
                        - 'efficientnet-b4' (lighter, often better on small datasets)
        num_classes:  Number of output segmentation classes.
        pretrained:   Load ImageNet weights for the encoder.

    Returns:
        model (nn.Module)
    """
    encoder_weights = "imagenet" if pretrained else None
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,          # raw logits; loss function applies softmax internally
    )
    return model
