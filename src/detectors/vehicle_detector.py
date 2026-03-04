from ultralytics import YOLO

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


class VehicleDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame, conf: float = 0.4):
        results = self.model(
            frame,
            conf=conf,
            classes=[2, 3, 5, 7],  # COCO vehicle classes
            verbose=False
        )

        vehicles = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                score = float(box.conf[0])

                vehicles.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "score": score
                })

        return vehicles