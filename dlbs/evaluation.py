from ultralytics import YOLO


def run_test_validation(
    weights_path,
    data_yaml: str,
    split: str = "test",
    *,
    imgsz=None,
    batch=None,
    device=None,
    workers=None,
    plots: bool = True,
    project: str | None = None,
    name: str | None = None,
):
    """Run validation on the requested split of ``data_yaml`` using the given weights."""
    eval_model = YOLO(str(weights_path))

    val_kwargs = {
        "data": data_yaml,
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "plots": plots,
        "project": project,
        "name": name,
        "exist_ok": True,
    }
    val_kwargs = {k: v for k, v in val_kwargs.items() if v is not None}
    return eval_model.val(**val_kwargs)
