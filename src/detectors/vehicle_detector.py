import yaml
from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, config_path: str):
        self.cfg = self._load_config(config_path)

        self.model = YOLO(self.cfg["model"]["path"])

        self.conf = self.cfg["model"].get("conf", 0.6)
        self.device = self.cfg["model"].get("device", "cpu")

        # Extract class IDs
        self.class_map = self.cfg["detection"]["class_map"]
        self.class_ids = list(self.class_map.values())

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.conf,
            classes=self.class_ids,
            device=self.device,
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
                    "class_name": self._class_name(cls_id),
                    "score": score
                })

        return vehicles

    def _class_name(self, cls_id: int) -> str:
        for name, cid in self.class_map.items():
            if cid == cls_id:
                return name
        return "unknown"