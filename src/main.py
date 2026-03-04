import cv2

from detectors.vehicle_detector import VehicleDetector
from recognizers.fastalpr_engine import FastALPREngine
from pipeline.alpr_pipeline import ALPRPipeline
from utils.draw import draw_results


def main():
    detector = VehicleDetector("configs/yolo.yaml")
    alpr = FastALPREngine()

    pipeline = ALPRPipeline(detector, alpr)

    cap = cv2.VideoCapture("assets/demo.mp4")  # webcam / video / RTSP
    width,height, fps  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS))
    vidWriter = cv2.VideoWriter("demo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    roi_x1, roi_y1, roi_x2, roi_y2 = 0, int(height*2/4), width, int(    height*3/4)  # Example ROI (middle horizontal band)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ## ROI Detection + ALPR
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        frame = cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)  # Draw ROI
        results = pipeline.process(roi_frame)
        frame[roi_y1:roi_y2, roi_x1:roi_x2] = draw_results(roi_frame, results)
        vidWriter.write(frame)
    cap.release()
    vidWriter.release()
if __name__ == "__main__":
    main()
