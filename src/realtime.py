import cv2
import torch
from model import DETR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from utils.rich_handlers import DetectionHandler
import time

# ================== LOGGER ==================
logger = get_logger("realtime")
detection_handler = DetectionHandler()

logger.print_banner()
logger.realtime("Initializing real-time sign language detection...")

# ================== TRANSFORMS ==================
transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ================== MODEL ==================
model = DETR(num_classes=3)
model.eval()
model.load_pretrained('pretrained/4426_model.pt')

CLASSES = get_classes()
COLORS = get_colors()

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)

# ================== UI FUNCTIONS ==================

# Clean bounding box (NO TEXT)
def draw_ui_box(frame, x1, y1, x2, y2, color):
    overlay = frame.copy()

    # Soft transparent fill
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame


# FPS + inference (top-left)
def draw_fps(frame, fps, inference_time):
    fps_text = f"FPS: {fps:.1f}"
    inf_text = f"{inference_time:.0f} ms"

    # Shadow
    cv2.putText(frame, fps_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 4)

    cv2.putText(frame, inf_text, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 4)

    # Text
    cv2.putText(frame, fps_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)

    cv2.putText(frame, inf_text, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 200, 255), 2)

    return frame


# ONLY place where class name is shown
def draw_summary(frame, detections):
    h, w, _ = frame.shape

    y = 40
    for det in detections[:3]:
        text = f"{det['class']}  {det['confidence']:.2f}"

        # Shadow
        cv2.putText(frame, text, (w - 300, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4)

        # Text
        cv2.putText(frame, text, (w - 300, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        y += 40

    return frame


# ================== PERFORMANCE ==================
frame_count = 0
fps = 0
fps_start_time = time.time()

# ================== WINDOW ==================
cv2.namedWindow('Sign Language Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Sign Language Detection', 1280, 720)

# ================== LOOP ==================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Smooth dark cinematic tone
    frame = cv2.convertScaleAbs(frame, alpha=0.95, beta=-15)

    # ================== INFERENCE ==================
    start = time.time()

    transformed = transforms(image=frame)
    input_tensor = torch.unsqueeze(transformed['image'], dim=0)

    result = model(input_tensor)

    inference_time = (time.time() - start) * 1000

    probabilities = result['pred_logits'].softmax(-1)[:, :, :-1]
    max_probs, max_classes = probabilities.max(-1)
    keep_mask = max_probs > 0.8

    batch_indices, query_indices = torch.where(keep_mask)

    bboxes = rescale_bboxes(
        result['pred_boxes'][batch_indices, query_indices, :],
        (frame.shape[1], frame.shape[0])
    )

    classes = max_classes[batch_indices, query_indices]
    probas = max_probs[batch_indices, query_indices]

    detections = []

    # ================== DRAW ==================
    for bclass, bprob, bbox in zip(classes, probas, bboxes):
        idx = bclass.item()
        conf = bprob.item()
        x1, y1, x2, y2 = map(int, bbox.detach().numpy())

        detections.append({
            'class': CLASSES[idx],
            'confidence': conf
        })

        # Draw ONLY box (no label)
        frame = draw_ui_box(frame, x1, y1, x2, y2, COLORS[idx])

    # ================== FPS ==================
    frame_count += 1
    if frame_count % 30 == 0:
        elapsed = time.time() - fps_start_time
        fps = 30 / elapsed
        fps_start_time = time.time()

    # ================== UI ==================
    frame = draw_fps(frame, fps, inference_time)
    frame = draw_summary(frame, detections)

    # ================== DISPLAY ==================
    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.realtime("Stopping detection...")
        break

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()