class ALPRPipeline:
    def __init__(self, vehicle_detector, alpr_engine):
        self.vehicle_detector = vehicle_detector
        self.alpr_engine = alpr_engine

    def process(self, frame):
        outputs = []
        vehicles = self.vehicle_detector.detect(frame)

        for v in vehicles:
            x1, y1, x2, y2 = v["bbox"]
            vehicle_crop = frame[y1:y2, x1:x2]

            if vehicle_crop.size == 0:
                continue

            plates = self.alpr_engine.recognize(vehicle_crop)

            outputs.append({
                "vehicle_bbox": v["bbox"],
                "vehicle_score": v["score"],
                "plates": plates
            })

        return outputs