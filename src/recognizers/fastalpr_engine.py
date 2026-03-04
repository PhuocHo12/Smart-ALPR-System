from fast_alpr import ALPR


class FastALPREngine:
    def __init__(self, detector="yolo-v9-t-384-license-plate-end2end", ocr="cct-xs-v1-global-model"):
        self.alpr = ALPR(
                detector_model=detector,
                ocr_model=ocr,
            )

    def recognize(self, image, threshold=0.6):
        results = self.alpr.predict(image)

        plates = []
        for r in results:
            conf = r.detection.confidence
            if conf < threshold:
                continue
            plates.append({
                "plate": r.ocr.text,
                "confidence": conf,
                "bbox": r.detection.bounding_box
            })

        return plates
