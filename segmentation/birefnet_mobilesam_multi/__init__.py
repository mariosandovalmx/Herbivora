"""Multi-leaf BiRefNet + MobileSAM segmentation (connected-component instances).

Activated only when Project → "Multiple leaves per photo" is checked.
Does not modify the single-leaf pipeline under birefnet_mobilesam/.
"""

__all__ = ["process_image_multi", "run_folder_batch_multi"]


def __getattr__(name: str):
    if name in __all__:
        from .run_multi_pipeline import process_image_multi, run_folder_batch_multi

        return {
            "process_image_multi": process_image_multi,
            "run_folder_batch_multi": run_folder_batch_multi,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
