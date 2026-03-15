from pathlib import Path
from ultralytics import YOLO

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = "yolov5s.pt"
    model = YOLO(str(model_path))

    exported_path = Path(
        model.export(
                 format="onnx", 
                 int8 = False,
                 dynamic = True,
                 simplify = True,
                 verbose = True,
                 opset = 12
                 ))
    target_path = models_dir / exported_path.name
    if exported_path.resolve() != target_path.resolve():
        exported_path.replace(target_path)
    print(f"ONNX model saved to: {target_path}")

if __name__ == "__main__":    
    main()
