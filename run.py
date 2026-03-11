import cv2
import numpy as np
import time
import os
import threading
import queue
import glob
from playsound import playsound
from PIL import Image, ImageDraw, ImageFont

# ================== AUTO MODE (Docker vs Local) ==================
IN_DOCKER = os.path.exists("/.dockerenv")

EXPORT_AUDIO = IN_DOCKER
OUT_AUDIO_DIR = "out_audio"
EVENT_LOG_PATH = os.path.join(OUT_AUDIO_DIR, "events.txt")

SHOW_WINDOW = False if IN_DOCKER else True
WINDOW_NAME = "Traffic Sign - ONNX + Vietnamese (Pre-generated Audio)"
# ================================================================

# ================== CONFIG ==================
MODEL_PATH = "best_640.onnx"
VIDEO_PATH = "input.mp4"
USE_CAMERA = False
CAMERA_ID = 0

IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45

# FPS target band (10-15)
FPS_MIN = 10.0
FPS_MAX = 15.0
# ===== FPS CAP (khóa tốc độ hiển thị/loop) =====
# Chọn một giá trị trong khoảng 10-15 để ổn định. Ví dụ 12 FPS.
TARGET_FPS = 12.0
FRAME_TIME = 1.0 / TARGET_FPS
# Infer frequency will be auto-adjusted to match FPS target
INFER_EVERY = 3
INFER_EVERY_MIN = 1
INFER_EVERY_MAX = 8

ASSETS_AUDIO_DIR = "assets_audio"

MAX_PLAY_QUEUE = 64

# Font config for Vietnamese

FONT_PATH = os.environ.get("VI_FONT_PATH", "assets/NotoSans-Regular.ttf")
FONT_SIZE = 16
# ===========================================

NAMES_EN = [
  'one way prohibition', 'no parking', 'no stopping and parking',
  'no turn left', 'no turn right', 'no u turn', 'no u and left turn',
  'no u and right turn', 'no motorbike entry/turning', 'no car entry/turning',
  'no truck entry/turning', 'other prohibition', 'indication', 'direction',
  'speed limit', 'weight limit', 'height limit', 'pedestrian crossing',
  'intersection danger', 'road danger', 'pedestrian danger', 'construction danger',
  'slow warning', 'other warning', 'vehicle permission lane',
  'vehicle and speed permission lane', 'overpass route', 'no more prohibition',
  'other'
]
OTHER_ID = 28

NAMES_VI = {
    0:  "Cấm đi ngược chiều",
    1:  "Cấm đỗ xe",
    2:  "Cấm dừng và đỗ",
    3:  "Cấm rẽ trái",
    4:  "Cấm rẽ phải",
    5:  "Cấm quay đầu",
    6:  "Cấm quay đầu và rẽ trái",
    7:  "Cấm quay đầu và rẽ phải",
    8:  "Cấm xe máy",
    9:  "Cấm ô tô",
    10: "Cấm xe tải",
    11: "Biển cấm khác",
    12: "Biển hiệu lệnh",
    13: "Biển chỉ dẫn hướng đi",
    14: "Giới hạn tốc độ",
    15: "Giới hạn tải trọng",
    16: "Giới hạn chiều cao",
    17: "Người đi bộ qua đường",
    18: "Cảnh báo giao nhau",
    19: "Cảnh báo nguy hiểm đường",
    20: "Cảnh báo người đi bộ",
    21: "Cảnh báo công trường",
    22: "Cảnh báo đi chậm",
    23: "Cảnh báo khác",
    24: "Làn đường cho phép",
    25: "Làn đường và tốc độ cho phép",
    26: "Hướng đi qua cầu vượt",
    27: "Hết cấm",
    28: "Khác"
}

# ======= ƯU TIÊN ĐỌC =======
PRIORITY = {i: 10 for i in range(29)}
for i in [18, 19, 20, 21, 22, 23]:
    PRIORITY[i] = 90
for i in list(range(0, 12)) + [27]:
    PRIORITY[i] = 80
for i in [14, 15, 16, 17]:
    PRIORITY[i] = 70
for i in [12, 13]:
    PRIORITY[i] = 50
for i in [24, 25, 26]:
    PRIORITY[i] = 40
PRIORITY[OTHER_ID] = -1

# =============== AUDIO ASSET INDEX (BUILD ONCE) ===============
def build_asset_index():
    idx = {}
    pattern = os.path.join(ASSETS_AUDIO_DIR, "*.mp3")
    for p in glob.glob(pattern):
        base = os.path.basename(p)
        try:
            cid = int(base.split("_", 1)[0])
            idx[cid] = p
        except Exception:
            continue
    return idx

play_q = queue.Queue(maxsize=MAX_PLAY_QUEUE)
spoken_ids = set()

def log_event(cid: int, text: str):
    os.makedirs(OUT_AUDIO_DIR, exist_ok=True)
    with open(EVENT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{time.time():.3f}\t{cid:02d}\t{text}\n")

def audio_worker():
    while True:
        item = play_q.get()
        if item is None:
            break
        try:
            cid, text, asset_path = item
            if EXPORT_AUDIO:
                log_event(cid, text)
            else:
                playsound(asset_path)
        except Exception as e:
            print("AUDIO error:", e)
        finally:
            play_q.task_done()

threading.Thread(target=audio_worker, daemon=True).start()

# ================== YOLOv8 ONNX ==================
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# (tùy máy) giảm overhead OpenCV threads đôi khi ổn định FPS hơn
try:
    cv2.setNumThreads(1)
except Exception:
    pass

def letterbox(im, size=640, color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(h * r), int(w * r)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    im = cv2.copyMakeBorder(im, top, top, left, left, cv2.BORDER_CONSTANT, value=color)
    return im, r, left, top

def nms_xywh(boxes, scores, iou):
    idxs = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou)
    return idxs.flatten().tolist() if len(idxs) else []

def choose_sign_to_speak(boxes, confs, cls_ids):
    best = None
    best_key = None
    for box, conf, cid in zip(boxes, confs, cls_ids):
        if cid == OTHER_ID:
            continue
        if cid in spoken_ids:
            continue
        pr = PRIORITY.get(cid, 10)
        if pr < 0:
            continue
        area = float(box[2] * box[3])
        key = (pr, area, float(conf))
        if best_key is None or key > best_key:
            best_key = key
            best = cid
    if best is None:
        return None
    return best, NAMES_VI.get(best, NAMES_EN[best])

# ================== VIETNAMESE TEXT RENDERING ==================
def load_vi_font():
    # Ưu tiên font TTF do bạn cung cấp (NotoSans, DejaVuSans, Roboto…)
    if os.path.exists(FONT_PATH):
        try:
            return ImageFont.truetype(FONT_PATH, FONT_SIZE)
        except Exception:
            pass
    # fallback: PIL default (không đảm bảo tiếng Việt)
    return ImageFont.load_default()

VI_FONT = load_vi_font()

def put_text_vi_bgr(frame_bgr, text, org, font=VI_FONT, fill=(0, 255, 0)):
    """
    Vẽ text Unicode (tiếng Việt) lên frame OpenCV bằng PIL.
    org: (x, y) top-left
    fill: màu (R,G,B)
    """
    # OpenCV BGR -> PIL RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)

    x, y = org
    # thêm nền đen mờ để dễ đọc
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rectangle((bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad), fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)

    # PIL RGB -> OpenCV BGR
    out_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return out_bgr

# ================== FPS CONTROL (AUTO THROTTLE) ==================
def adjust_infer_every(current_infer_every, fps_avg):
    """
    Nếu FPS < FPS_MIN: tăng INFER_EVERY (infer ít hơn) để FPS tăng.
    Nếu FPS > FPS_MAX: giảm INFER_EVERY (infer nhiều hơn) để FPS giảm về band.
    """
    if fps_avg < FPS_MIN and current_infer_every < INFER_EVERY_MAX:
        return current_infer_every + 1
    if fps_avg > FPS_MAX and current_infer_every > INFER_EVERY_MIN:
        return current_infer_every - 1
    return current_infer_every

def main():
    if not os.path.isdir(ASSETS_AUDIO_DIR):
        raise RuntimeError(
            f"Không thấy thư mục '{ASSETS_AUDIO_DIR}'. "
            f"Hãy generate audio trước vào assets_audio/."
        )

    ASSET_INDEX = build_asset_index()
    if not ASSET_INDEX:
        raise RuntimeError("assets_audio/ không có mp3 hoặc format tên file không đúng (vd: 01_xxx.mp3).")

    cap = cv2.VideoCapture(CAMERA_ID if USE_CAMERA else VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Không mở được video/camera. Kiểm tra đường dẫn hoặc camera id.")

    last_boxes, last_confs, last_cls = [], [], []
    fps_hist = []

    # Dùng perf_counter cho đo thời gian chính xác hơn time.time()
    prev = time.perf_counter()
    fid = 0

    infer_every = INFER_EVERY
    last_adjust_t = time.perf_counter()

    # FPS CAP: lên lịch frame tiếp theo
    next_frame_time = time.perf_counter()

    while True:
        # -------- FPS CAP: giữ nhịp loop trước khi đọc frame --------
        # (cap ở đầu vòng lặp thường ổn định hơn)
        now_perf = time.perf_counter()
        sleep_t = next_frame_time - now_perf
        if sleep_t > 0:
            time.sleep(sleep_t)
        next_frame_time += FRAME_TIME

        # Nếu bị lag quá lâu, reset để tránh drift tích lũy
        if now_perf - next_frame_time > 0.5:
            next_frame_time = now_perf + FRAME_TIME
        # ------------------------------------------------------------

        cap.grab()
        ret, frame = cap.read()
        if not ret:
            break

        fid += 1
        H, W = frame.shape[:2]

        # --- INFERENCE (throttle by infer_every) ---
        if fid % infer_every == 0:
            img, r, dw, dh = letterbox(frame, IMG_SIZE)
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (IMG_SIZE, IMG_SIZE), swapRB=True, crop=False)
            net.setInput(blob)
            out = net.forward()[0]  # (4+nc, N)

            boxes_raw = out[:4, :]
            scores_raw = out[4:, :]

            conf_all = scores_raw.max(axis=0)
            cls_all = scores_raw.argmax(axis=0)

            boxes, confs, cls_ids = [], [], []
            for i in range(conf_all.shape[0]):
                conf = float(conf_all[i])
                if conf < CONF_THRES:
                    continue
                cid = int(cls_all[i])
                if cid == OTHER_ID:
                    continue

                cx, cy, bw, bh = boxes_raw[:, i]
                x1 = (cx - bw/2 - dw) / r
                y1 = (cy - bh/2 - dh) / r
                x2 = (cx + bw/2 - dw) / r
                y2 = (cy + bh/2 - dh) / r

                x1 = max(0, min(W, x1)); y1 = max(0, min(H, y1))
                x2 = max(0, min(W, x2)); y2 = max(0, min(H, y2))

                w_box = max(0, int(x2 - x1))
                h_box = max(0, int(y2 - y1))
                if w_box == 0 or h_box == 0:
                    continue

                boxes.append([int(x1), int(y1), w_box, h_box])
                confs.append(conf)
                cls_ids.append(cid)

            keep = nms_xywh(boxes, confs, IOU_THRES)
            last_boxes = [boxes[i] for i in keep]
            last_confs = [confs[i] for i in keep]
            last_cls   = [cls_ids[i] for i in keep]

            chosen = choose_sign_to_speak(last_boxes, last_confs, last_cls)
            if chosen is not None:
                cid, text = chosen
                asset = ASSET_INDEX.get(cid)
                if asset is None:
                    print(f"[WARN] Missing audio asset for cid={cid} text='{text}'.")
                else:
                    try:
                        play_q.put_nowait((cid, text, asset))
                        spoken_ids.add(cid)
                    except queue.Full:
                        pass

        # --- FPS measurement ---
        now = time.perf_counter()
        fps = 1.0 / max(1e-6, (now - prev))
        prev = now
        fps_hist.append(fps)
        if len(fps_hist) > 30:
            fps_hist.pop(0)
        fps_avg = float(np.mean(fps_hist))

        # Auto-adjust infer_every (tùy chọn: bạn vẫn có thể giữ phần này)
        if (now - last_adjust_t) >= 1.0:
            infer_every = adjust_infer_every(infer_every, fps_avg)
            last_adjust_t = now

        # --- DISPLAY ---
        if SHOW_WINDOW:
            for (x, y, w, h), conf, cid in zip(last_boxes, last_confs, last_cls):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{NAMES_VI.get(cid, NAMES_EN[cid])} ({conf:.2f})"
                frame = put_text_vi_bgr(frame, label, (x, max(0, y - 24)), fill=(0, 255, 0))

            # Overlay: chỉ hiện FPS (KHÔNG hiện INFER_EVERY)
            frame = put_text_vi_bgr(
                frame,
                f"FPS: {fps_avg:.1f}",
                (10, 10),
                fill=(255, 255, 255)
            )

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    play_q.join()
    play_q.put(None)

    if EXPORT_AUDIO:
        print(f"Done. Docker mode: wrote events to ./{EVENT_LOG_PATH} (không copy mp3 để tránh chậm).")
    else:
        print("Done.")

if __name__ == "__main__":
    main()
