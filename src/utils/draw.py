import cv2


def draw_results(frame, results):
    for r in results:
        x1, y1, x2, y2 = r["vehicle_bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for p in r["plates"]:
            text = f"{p['plate']} ({p['confidence']:.2f})"
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
    return frame