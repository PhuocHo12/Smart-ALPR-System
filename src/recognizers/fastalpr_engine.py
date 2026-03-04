import yaml
from fast_alpr import ALPR


class FastALPREngine:
    def __init__(self, config_path: str):
        self.cfg = self._load_config(config_path)["alpr"]

        self.detector_model = self.cfg["detector_model"]
        self.ocr_model = self.cfg["ocr_model"]
        self.threshold = self.cfg.get("threshold", 0.6)

        self.alpr = ALPR(
            detector_model=self.detector_model,
            ocr_model=self.ocr_model,
        )

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def recognize(self, image, threshold: float | None = None):
        results = self.alpr.predict(image)

        plates = []
        for r in results:
            conf = r.detection.confidence

            if conf < (threshold if threshold is not None else self.threshold):
                continue

            plates.append({
                "plate": r.ocr.text,
                "confidence": conf,
                "bbox": r.detection.bounding_box
            })

        return plates