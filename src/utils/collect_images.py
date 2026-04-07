import cv2
import numpy as np
import uuid
import time
import os
from setup import get_classes
from logger import logger

classes = get_classes()

class CaptureImages():
    def __init__(self, path: str, classes: dict, camera_id: int) -> None:
        self.cap = cv2.VideoCapture(camera_id)
        self.path = path
        self.classes = classes

        logger.print_banner()
        logger.capture("Image capture system initialized")

        if not self.cap.isOpened():
            logger.capture_error("Camera", f"Could not open camera {camera_id}")
            raise Exception(f"Could not open camera {camera_id}")
        else:
            logger.success(f"Camera {camera_id} connected successfully")

        os.makedirs(self.path, exist_ok=True)
        logger.info(f"Output directory: {self.path}")

    # ================= UI FUNCTIONS =================

    def draw_ui(self, frame, class_name, count, total):
        h, w, _ = frame.shape

        # Dark cinematic tone
        frame = cv2.convertScaleAbs(frame, alpha=0.95, beta=-15)

        # ===== Title (Top Center) =====
        title = f"Capturing: {class_name}"

        cv2.putText(frame, title, (int(w/2)-200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 4)

        cv2.putText(frame, title, (int(w/2)-200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        # ===== Progress Text =====
        progress_text = f"{count}/{total}"

        cv2.putText(frame, progress_text, (int(w/2)-50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 0), 5)

        cv2.putText(frame, progress_text, (int(w/2)-50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255), 2)

        # ===== Progress Bar =====
        bar_width = 400
        bar_x = int((w - bar_width) / 2)
        bar_y = 130

        progress = int((count / total) * bar_width)

        # Background line
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + 20),
                      (80, 80, 80), -1)

        # Progress fill
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + progress, bar_y + 20),
                      (0, 255, 100), -1)

        return frame

    # Flash animation when image captured
    def flash_effect(self, frame):
        white = np.ones_like(frame) * 255
        frame = cv2.addWeighted(frame, 0.5, white, 0.5, 0)
        return frame

    # ================= CAPTURE =================

    def capture(self, class_name: str, count, total) -> bool:
        try:
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Failed to read from camera")

            raw_frame = frame.copy()

            # Draw UI
            frame = self.draw_ui(frame, class_name, count, total)

            cv2.imshow('Image Capture', frame)

            # Save image
            filename = f'{class_name}-{uuid.uuid1()}.jpg'
            filepath = os.path.join(self.path, filename)
            cv2.imwrite(filepath, raw_frame)

            # Flash effect
            flash = self.flash_effect(frame)
            cv2.imshow('Image Capture', flash)
            cv2.waitKey(50)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.warning("Quit key pressed")
                return False

            return True

        except Exception as e:
            logger.capture_error(class_name, str(e))
            return False

    # ================= RUN =================

    def run(self, sleep_time: int = 1, num_images: int = 10):
        logger.capture_session_start(self.classes, num_images, sleep_time)

        total_captured = 0

        for img_class in self.classes:
            logger.capture_class_start(img_class, num_images)

            class_captured = 0

            for idx in range(num_images):
                success = self.capture(img_class, idx + 1, num_images)

                if success:
                    class_captured += 1
                    total_captured += 1
                    logger.capture_success(img_class, idx + 1)
                else:
                    logger.capture_error(img_class, f"Image {idx + 1}")

                time.sleep(sleep_time)

            logger.success(f"{img_class}: {class_captured}/{num_images}")

            time.sleep(10)

        logger.capture_session_complete(total_captured, len(self.classes))

        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released")


# ================= MAIN =================

if __name__ == '__main__':
    cap = CaptureImages('./data/test/', classes, 0)
    cap.run(num_images=5)