import argparse
import cv2
import ctypes
import numpy as np
import math
import os
import json
import re
import shutil
import sys
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog
from ultralytics import YOLO


def activate_english_keyboard_layout():
    """在 Windows OpenCV 視窗開啟前切換到英文鍵盤配置。"""
    if os.name != "nt":
        return
    try:
        user32 = ctypes.windll.user32
        english_layout = user32.LoadKeyboardLayoutW("00000409", 0x00000001)
        if english_layout:
            user32.ActivateKeyboardLayout(english_layout, 0)
    except (AttributeError, OSError):
        pass


def apply_english_keyboard_layout_to_window(window_name):
    """將英文鍵盤配置套用到指定的 OpenCV 視窗。"""
    activate_english_keyboard_layout()
    cv2.waitKey(1)
    if os.name != "nt":
        return
    try:
        user32 = ctypes.windll.user32
        english_layout = user32.LoadKeyboardLayoutW("00000409", 0x00000001)
        window_handle = user32.FindWindowW(None, window_name)
        if english_layout and window_handle:
            user32.PostMessageW(window_handle, 0x0050, 0, english_layout)
    except (AttributeError, OSError):
        pass


def maximize_cv_window(window_name):
    """將 OpenCV 視窗最大化到目前 Windows 螢幕。"""
    cv2.waitKey(1)
    if os.name != "nt":
        return
    try:
        window_handle = ctypes.windll.user32.FindWindowW(None, window_name)
        if window_handle:
            ctypes.windll.user32.ShowWindow(window_handle, 3)
    except (AttributeError, OSError):
        pass


def screen_size():
    """取得目前螢幕大小，非 Windows 環境使用保守的預設值。"""
    if os.name == "nt":
        try:
            user32 = ctypes.windll.user32
            return max(640, user32.GetSystemMetrics(0)), max(480, user32.GetSystemMetrics(1))
        except (AttributeError, OSError):
            pass
    return 1920, 1080


def select_video_file():
    """開啟原生系統檔案選擇器，讓使用者自由選擇影片"""
    # 建立隱藏的 Tkinter 根視窗，避免彈出空白背景框
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # 強制讓檔案選擇器出現在最前面

    print("請在彈出的視窗中選擇要匯入的遙控甩尾影片...")
    file_path = filedialog.askopenfilename(
        title="選擇遙控甩尾車影片",
        filetypes=[
            ("影片檔案", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("所有檔案", "*.*")
        ]
    )
    
    root.destroy()
    return file_path


def select_video_files():
    """選擇多支影片，供自動擴充資料集使用。"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    print("請在彈出的視窗中選擇要自動擴充訓練資料的影片（可多選）...")
    file_paths = filedialog.askopenfilenames(
        title="選擇多支遙控甩尾車影片",
        filetypes=[
            ("影片檔案", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("所有檔案", "*.*")
        ]
    )
    root.destroy()
    return list(file_paths)


def _merge_overlapping_car_polygons(candidates, frame_shape, merge_iou=0.35,
                                     merge_center_ratio=0.55, merge_min_distance=30.0,
                                     merge_containment=0.65):
    """將重疊/相鄰的候選遮罩視為同一台車，合併遮罩後回傳單一輪廓（與記分模式合併邏輯一致）。"""
    count = len(candidates)
    if count == 0:
        return []
    boxes = [item[1] for item in candidates]
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for i in range(count):
        for j in range(i + 1, count):
            box, other = boxes[i], boxes[j]
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            other_area = max(0.0, other[2] - other[0]) * max(0.0, other[3] - other[1])
            intersection = (
                max(0.0, min(box[2], other[2]) - max(box[0], other[0])) *
                max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
            )
            union_area = area + other_area - intersection
            smaller = max(min(area, other_area), 1.0)
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            other_center = ((other[0] + other[2]) / 2, (other[1] + other[3]) / 2)
            center_distance = math.hypot(center[0] - other_center[0], center[1] - other_center[1])
            size_limit = max(
                merge_min_distance,
                min(box[2] - box[0], box[3] - box[1],
                    other[2] - other[0], other[3] - other[1]) * merge_center_ratio,
            )
            close_centers = center_distance <= size_limit
            if ((union_area > 0 and intersection / union_area >= merge_iou) or
                    intersection / smaller >= merge_containment or close_centers):
                union(i, j)

    groups = {}
    for index in range(count):
        root = find(index)
        groups.setdefault(root, []).append(index)

    height, width = frame_shape[:2]
    merged = []
    for indices in groups.values():
        if len(indices) == 1:
            confidence, _box, polygon = candidates[indices[0]]
            merged.append((confidence, polygon))
            continue
        canvas = np.zeros((height, width), dtype=np.uint8)
        best_confidence = 0.0
        for index in indices:
            confidence, _box, polygon = candidates[index]
            best_confidence = max(best_confidence, confidence)
            cv2.fillPoly(canvas, [polygon.astype(np.int32)], 255)
        contours, _hierarchy = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.002 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        merged_polygon = approx.reshape(-1, 2).astype(np.float32)
        if merged_polygon.shape[0] < 3:
            continue
        merged.append((best_confidence, merged_polygon))
    return merged


def auto_augment_vehicle_data(video_paths, epochs=50, sample_every=10, min_conf=0.12,
                              max_per_video=20, duplicate_threshold=0.0,
                              max_label_count=None, exact_label_count=None,
                              train_after_capture=False, inference_imgsz=1280,
                              max_candidates=20, merge_iou=0.35, merge_center_ratio=0.55,
                              merge_min_distance=30.0, merge_containment=0.65):
    """用現有車輛模型，對其他影片做自動偵測並產生 pseudo-label，快速擴充訓練資料。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "vehicle_seg_training")
    image_dir = os.path.join(dataset_dir, "images", "train")
    label_dir = os.path.join(dataset_dir, "labels", "train")
    processed_path = os.path.join(dataset_dir, "auto_augment_processed.json")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    try:
        with open(processed_path, "r", encoding="utf-8") as processed_file:
            processed_frames = set(json.load(processed_file))
    except (OSError, json.JSONDecodeError, TypeError):
        processed_frames = set()

    if not video_paths:
        print("[自動擴充] 沒有選擇任何影片。")
        return False

    latest_model = os.path.join(dataset_dir, "runs_latest", "weights", "best.pt")
    trained_model = os.path.join(base_dir, "rc_car_model.pt")
    if os.path.isfile(latest_model):
        model_path = latest_model
    elif os.path.isfile(trained_model):
        model_path = trained_model
    else:
        model_path = "yolov8n-seg.pt"
    print(f"[自動擴充] 使用模型: {model_path} | conf={min_conf}")
    model = YOLO(model_path)
    added_count = 0
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[自動擴充] 無法開啟影片: {video_path}")
            continue

        frame_index = 0
        saved_this_video = 0
        previous_saved_gray = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % max(1, int(sample_every)) != 0:
                continue
            frame_key = f"{os.path.normcase(os.path.abspath(video_path))}|{frame_index}"
            if frame_key in processed_frames:
                continue

            results = model(frame, imgsz=inference_imgsz, verbose=False, conf=min_conf,
                            iou=0.45, max_det=max_candidates)
            if not results:
                continue

            result = results[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                print(f"[自動擴充] 影片 {os.path.basename(video_path)} 在幀 {frame_index} 沒有 box 偵測，conf={min_conf}。")
                continue

            print(f"[自動擴充] 影片 {os.path.basename(video_path)} 在幀 {frame_index} 偵測到 {len(boxes)} 個 box。")

            if max_label_count is not None and len(boxes) > max_label_count:
                print(f"[自動擴充] 影片 {os.path.basename(video_path)} 幀 {frame_index} 超過 {max_label_count} 個候選，跳過保存。")
                continue
            if len(boxes) >= max_candidates:
                print(f"[自動擴充] 影片 {os.path.basename(video_path)} 幀 {frame_index} 達到候選上限 {max_candidates}，跳過保存。")
                continue

            if result.masks is None:
                print(f"[自動擴充] 影片 {os.path.basename(video_path)} 在幀 {frame_index} 只偵測到 box，沒有 mask，跳過。")
                continue

            if result.masks is not None and len(result.masks.xy) > 0:
                masks = result.masks.xy
                candidates = []
                for idx, mask in enumerate(masks):
                    if idx >= len(boxes.cls):
                        continue
                    cls_id = int(boxes.cls[idx].item())
                    if cls_id not in (0,):
                        continue
                    polygon = np.asarray(mask, dtype=np.float32)
                    if polygon.shape[0] < 3:
                        continue
                    box = boxes.xyxy[idx].cpu().numpy().astype(np.float32)
                    confidence = float(boxes.conf[idx].item())
                    candidates.append((confidence, box, polygon))

                # 將重疊/相鄰、屬於同一台車的多個候選遮罩合併成單一輪廓（同記分模式邏輯）。
                merged_candidates = _merge_overlapping_car_polygons(
                    candidates, frame.shape,
                    merge_iou=merge_iou,
                    merge_center_ratio=merge_center_ratio,
                    merge_min_distance=merge_min_distance,
                    merge_containment=merge_containment,
                )
                valid_polygons = [polygon for _confidence, polygon in merged_candidates]

                if valid_polygons:
                    if (exact_label_count is not None and
                            len(valid_polygons) != exact_label_count):
                        print(f"[自動擴充] 影片 {os.path.basename(video_path)} 幀 {frame_index} 有 {len(valid_polygons)} 個標記，不是指定的 {exact_label_count} 個，跳過保存。")
                        continue
                    if max_label_count is not None and len(valid_polygons) > max_label_count:
                        print(f"[自動擴充] 影片 {os.path.basename(video_path)} 幀 {frame_index} 有 {len(valid_polygons)} 個標記，超過上限 {max_label_count}，跳過保存。")
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    comparison = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
                    is_duplicate = (duplicate_threshold > 0 and previous_saved_gray is not None and
                                    float(cv2.absdiff(comparison, previous_saved_gray).mean()) < duplicate_threshold)
                    if is_duplicate:
                        continue

                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_path = os.path.join(image_dir, f"car_{stamp}.jpg")
                    label_path = os.path.join(label_dir, f"car_{stamp}.txt")
                    if not cv2.imwrite(image_path, frame):
                        continue
                    with open(label_path, "w", encoding="utf-8") as label_file:
                        height, width = frame.shape[:2]
                        for polygon in valid_polygons:
                            normalized = polygon.copy()
                            normalized[:, 0] /= max(1, width)
                            normalized[:, 1] /= max(1, height)
                            coords = " ".join(f"{value:.6f}" for value in normalized.flatten())
                            label_file.write(f"0 {coords}\n")
                    processed_frames.add(frame_key)
                    with open(processed_path, "w", encoding="utf-8") as processed_file:
                        json.dump(sorted(processed_frames), processed_file, ensure_ascii=False, indent=2)
                    previous_saved_gray = comparison
                    saved_this_video += 1
                    added_count += 1
                if saved_this_video >= max_per_video:
                    break

        cap.release()
        print(f"[自動擴充] 影片 {os.path.basename(video_path)} 新增 {saved_this_video} 張樣本。")

    if added_count == 0:
        print("[自動擴充] 沒有任何影片被成功自動標註，請確認模型品質或調高 min_conf。")
        return False

    print(f"[自動擴充] 總共新增 {added_count} 張自動標註樣本。")
    if train_after_capture:
        print("[自動擴充] 已啟用訓練，開始重新訓練車輛模型...")
        return _train_vehicle_model(dataset_dir, epochs)
    print("[自動擴充] 目前只保存自動標註，不進行訓練。")
    return True


def collect_vehicle_training_data(video_path, epochs=50, sample_every=10, train_after_save=False):
    """從影片手動框選遙控車，將人工標記加入訓練集。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "vehicle_seg_training")
    image_dir = os.path.join(dataset_dir, "images", "train")
    label_dir = os.path.join(dataset_dir, "labels", "train")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[錯誤] 無法開啟車輛訓練影片。")
        return

    window_name = "Vehicle Training - draw RC car outlines"
    polygons = []
    current_polygon = []
    current_frame = None
    display_scale = 1.0

    def mouse_callback(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append((int(x / display_scale), int(y / display_scale)))

    activate_english_keyboard_layout()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    apply_english_keyboard_layout_to_window(window_name)
    maximize_cv_window(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    frame_index = 0
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % max(1, sample_every) != 0:
            continue

        current_frame = frame.copy()
        polygons.clear()
        current_polygon.clear()
        height, width = frame.shape[:2]
        display_width = min(1600, width)
        display_scale = display_width / width
        display_height = max(1, int(height * display_scale))

        while True:
            display = cv2.resize(current_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
            for polygon in polygons + ([current_polygon] if current_polygon else []):
                points = (np.asarray(polygon, dtype=np.float32) * display_scale).astype(np.int32)
                cv2.polylines(display, [points], len(polygon) >= 3, (0, 255, 255), 4, cv2.LINE_AA)
                for point in points:
                    cv2.circle(display, tuple(point), 6, (0, 165, 255), -1)
            cv2.rectangle(display, (8, 8), (display_width - 8, 142), (0, 0, 0), -1)
            cv2.putText(display, "MOUSE LEFT: add outline points | M: finish this car",
                        (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "N / ENTER: save frame | U: undo | C: clear current outline",
                        (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "S: skip frame | Q: finish collection",
                        (18, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"Saved: {saved_count} | Cars: {len(polygons)} | Points: {len(current_polygon)}",
                        (18, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("u") and current_polygon:
                current_polygon.pop()
            elif key == ord("c"):
                current_polygon.clear()
            elif key == ord("s"):
                break
            elif key == ord("q"):
                cap.release()
                cv2.destroyWindow(window_name)
                print(f"[車輛資料] 已保存 {saved_count} 張，停止收集。")
                if train_after_save:
                    return _train_vehicle_model(dataset_dir, epochs)
                return True
            elif key == ord("m") and len(current_polygon) >= 3:
                polygons.append(current_polygon.copy())
                current_polygon.clear()
            elif key in (ord("n"), 13):
                if len(current_polygon) >= 3:
                    polygons.append(current_polygon.copy())
                    current_polygon.clear()
                if not polygons:
                    print("[車輛資料] 先描繪至少一個完整輪廓，再按 N 保存。")
                    continue
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                image_path = os.path.join(image_dir, f"car_{stamp}.jpg")
                label_path = os.path.join(label_dir, f"car_{stamp}.txt")
                cv2.imwrite(image_path, current_frame)
                with open(label_path, "w", encoding="utf-8") as label_file:
                    for polygon in polygons:
                        normalized = np.asarray(polygon, dtype=np.float32)
                        normalized[:, 0] /= width
                        normalized[:, 1] /= height
                        coordinates = " ".join(f"{value:.6f}" for value in normalized.flatten())
                        label_file.write(f"0 {coordinates}\n")
                saved_count += 1
                print(f"[車輛資料] 已保存第 {saved_count} 張輪廓：{os.path.basename(image_path)}")
                break

    cap.release()
    cv2.destroyWindow(window_name)
    if train_after_save:
        return _train_vehicle_model(dataset_dir, epochs)
    print(f"[車輛資料] 收集完成，共保存 {saved_count} 張；目前不訓練模型。")
    return True


def _parse_images_after(value):
    """解析圖片時間篩選門檻；支援檔名時間與一般日期時間格式。"""
    if not value:
        return None
    for date_format in ("%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d_%H-%M-%S"):
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            continue
    raise ValueError("時間格式應為 YYYYMMDD_HHMMSS，例如 20260918_010000")


def _image_is_after(image_name, image_path, after_time):
    """以檔名時間判定新增時間；無法解析時退回檔案修改時間。"""
    if after_time is None:
        return True
    timestamp_match = re.search(r"(20\d{6}_\d{6})", image_name)
    if timestamp_match:
        image_time = datetime.strptime(timestamp_match.group(1), "%Y%m%d_%H%M%S")
        return image_time > after_time
    try:
        return datetime.fromtimestamp(os.path.getmtime(image_path)) > after_time
    except OSError:
        return False


def curate_validation_data(label_count=None, unseen_only=True, images_after=None):
    """逐張檢視指定標註數量的訓練圖片，移到驗證集。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "vehicle_seg_training")
    train_image_dir = os.path.join(dataset_dir, "images", "train")
    train_label_dir = os.path.join(dataset_dir, "labels", "train")
    val_image_dir = os.path.join(dataset_dir, "images", "val")
    val_label_dir = os.path.join(dataset_dir, "labels", "val")
    reviewed_path = os.path.join(dataset_dir, "curate_reviewed.json")
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    try:
        with open(reviewed_path, "r", encoding="utf-8") as reviewed_file:
            reviewed_names = set(json.load(reviewed_file))
    except (OSError, json.JSONDecodeError, TypeError):
        reviewed_names = set()

    all_image_names = sorted(
        name for name in os.listdir(train_image_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    all_image_names = [
        name for name in all_image_names
        if _image_is_after(name, os.path.join(train_image_dir, name), images_after)
    ]
    if unseen_only:
        all_image_names = [name for name in all_image_names if name not in reviewed_names]
    image_names = []
    for image_name in all_image_names:
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(train_label_dir, label_name)
        if not os.path.exists(label_path):
            continue
        with open(label_path, "r", encoding="utf-8") as label_file:
            valid_label_count = sum(
                1 for line in label_file
                if len(line.split()) >= 7 and (len(line.split()) - 1) % 2 == 0
            )
        if label_count is None or valid_label_count == label_count:
            image_names.append(image_name)

    if not image_names:
        filter_text = "所有標註數量" if label_count is None else f"標註數量為 {label_count}"
        print(f"[驗證集挑選] 找不到符合 {filter_text} 的圖片。")
        return False

    if not all_image_names:
        print("[驗證集挑選] train 資料夾沒有可挑選的圖片。")
        return False

    window_name = "Validation Curation - P: pickup | S: skip | D: delete | Q: quit"
    activate_english_keyboard_layout()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    apply_english_keyboard_layout_to_window(window_name)
    maximize_cv_window(window_name)
    picked_count = 0
    skipped_count = 0

    for index, image_name in enumerate(image_names, start=1):
        image_path = os.path.join(train_image_dir, image_name)
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(train_label_dir, label_name)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[驗證集挑選] 無法讀取，跳過: {image_name}")
            continue
        if not os.path.exists(label_path):
            print(f"[驗證集挑選] 找不到對應標註，跳過: {image_name}")
            continue

        height, width = frame.shape[:2]
        polygons = []
        with open(label_path, "r", encoding="utf-8") as label_file:
            for line in label_file:
                values = line.split()
                if len(values) < 7 or (len(values) - 1) % 2 != 0:
                    continue
                points = np.asarray([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
                polygons.append(points)
        original_polygons = [polygon.copy() for polygon in polygons]
        annotation_count = len(polygons)
        display_width = min(1280, width)
        scale = display_width / max(1, width)
        display_height = max(1, int(height * scale))
        edit_mode = False
        selected_vertex = None
        dragging = False

        def mouse_callback(event, x, y, flags, parameter):
            nonlocal selected_vertex, dragging
            if not edit_mode:
                return
            point = np.asarray([x / scale, y / scale], dtype=np.float32)
            if event == cv2.EVENT_LBUTTONDOWN:
                if y >= display_height:
                    return
                nearest = None
                nearest_distance = 20.0 / max(scale, 0.01)
                for polygon_index, polygon in enumerate(polygons):
                    for vertex_index, vertex in enumerate(
                            polygon * np.asarray([width, height], dtype=np.float32)):
                        distance = float(np.linalg.norm(vertex - point))
                        if distance < nearest_distance:
                            nearest = (polygon_index, vertex_index)
                            nearest_distance = distance
                selected_vertex = nearest
                dragging = nearest is not None
            elif event == cv2.EVENT_MOUSEMOVE and dragging and selected_vertex is not None:
                polygon_index, vertex_index = selected_vertex
                polygons[polygon_index][vertex_index] = (
                    min(width - 1, max(0, point[0] / width)),
                    min(height - 1, max(0, point[1] / height)),
                )
            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False

        cv2.setMouseCallback(window_name, mouse_callback)
        while True:
            display = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
            for polygon in polygons:
                points = (polygon * np.asarray([width, height], dtype=np.float32) * scale).astype(np.int32)
                cv2.polylines(display, [points], True, (0, 255, 0), 4, cv2.LINE_AA)
                for point in points:
                    cv2.circle(display, tuple(point), 7, (0, 200, 255), -1)
                center = tuple(np.mean(points, axis=0).astype(int))
                cv2.putText(display, "PREDICTED CAR", center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            panel_height = 100
            canvas = np.zeros((display_height + panel_height, display_width, 3), dtype=np.uint8)
            canvas[:display_height] = display
            edit_text = "EDIT ON: drag vertices | R: reset" if edit_mode else "E: edit vertices"
            cv2.putText(canvas, f"P: PICKUP | S: SKIP | D: DELETE | Q: QUIT | {edit_text}",
                        (16, display_height + 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{index}/{len(image_names)}  labels: {annotation_count}  picked: {picked_count}  skipped: {skipped_count}",
                        (16, display_height + 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("p"), ord("P")):
                with open(label_path, "w", encoding="utf-8") as label_file:
                    for polygon in polygons:
                        label_file.write("0 " + " ".join(f"{value:.6f}" for value in polygon.flatten()) + "\n")
                shutil.move(image_path, os.path.join(val_image_dir, image_name))
                shutil.move(label_path, os.path.join(val_label_dir, label_name))
                picked_count += 1
                reviewed_names.add(image_name)
                print(f"[驗證集挑選] 已移入驗證集: {image_name}")
                break
            if key in (ord("s"), ord("S")):
                # 略過但仍留在訓練集時，也要保存編輯後的 polygon，否則手動修正的形狀會遺失。
                with open(label_path, "w", encoding="utf-8") as label_file:
                    for polygon in polygons:
                        label_file.write("0 " + " ".join(f"{value:.6f}" for value in polygon.flatten()) + "\n")
                skipped_count += 1
                reviewed_names.add(image_name)
                break
            if key in (ord("d"), ord("D")):
                os.remove(image_path)
                os.remove(label_path)
                reviewed_names.add(image_name)
                print(f"[驗證集挑選] 已刪除圖片與標註: {image_name}")
                break
            if key in (ord("e"), ord("E")):
                edit_mode = not edit_mode
                selected_vertex = None
                dragging = False
            if key in (ord("r"), ord("R")):
                polygons = [polygon.copy() for polygon in original_polygons]
            if key in (ord("q"), ord("Q"), 27):
                with open(reviewed_path, "w", encoding="utf-8") as reviewed_file:
                    json.dump(sorted(reviewed_names), reviewed_file, ensure_ascii=False, indent=2)
                cv2.destroyWindow(window_name)
                print(f"[驗證集挑選] 結束：移入 {picked_count} 張，跳過 {skipped_count} 張。")
                return True

    cv2.destroyWindow(window_name)
    with open(reviewed_path, "w", encoding="utf-8") as reviewed_file:
        json.dump(sorted(reviewed_names), reviewed_file, ensure_ascii=False, indent=2)
    print(f"[驗證集挑選] 完成：移入 {picked_count} 張，跳過 {skipped_count} 張。")
    return True


def review_existing_labels(split="all", images_after=None):
    """檢查既有 train/val 標註，允許補畫漏標車輛。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "vehicle_seg_training")
    splits = ("train", "val") if split == "all" else (split,)
    samples = []
    for current_split in splits:
        image_dir = os.path.join(dataset_dir, "images", current_split)
        label_dir = os.path.join(dataset_dir, "labels", current_split)
        if os.path.isdir(image_dir):
            for name in sorted(os.listdir(image_dir)):
                image_path = os.path.join(image_dir, name)
                if (name.lower().endswith((".jpg", ".jpeg", ".png")) and
                        _image_is_after(name, image_path, images_after)):
                    samples.append((current_split, image_path,
                                    os.path.join(label_dir, os.path.splitext(name)[0] + ".txt")))
    if not samples:
        print("[標註檢查] 找不到圖片。")
        return False

    window_name = "Label Review - A: add | M: finish | E: edit vertex | Right-click: del polygon | D: del image | W: write | B: back | S: next | Q: quit"
    activate_english_keyboard_layout()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    apply_english_keyboard_layout_to_window(window_name)
    maximize_cv_window(window_name)
    sample_index = 0
    while sample_index < len(samples):
        current_split, image_path, label_path = samples[sample_index]
        frame = cv2.imread(image_path)
        if frame is None:
            sample_index += 1
            continue
        height, width = frame.shape[:2]
        polygons = []
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as label_file:
                for line in label_file:
                    values = line.split()
                    if len(values) >= 7 and (len(values) - 1) % 2 == 0:
                        polygons.append(np.asarray([float(v) for v in values[1:]], dtype=np.float32).reshape(-1, 2))
        adding = False
        current_points = []
        dirty = False
        edit_mode = False
        selected_vertex = None
        dragging = False

        while True:
            display_width = min(1280, width)
            scale = display_width / max(1, width)
            display_height = max(1, int(height * scale))
            display = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
            for polygon in polygons:
                points = (polygon * np.asarray([width, height], dtype=np.float32) * scale).astype(np.int32)
                cv2.polylines(display, [points], True, (0, 255, 0), 3, cv2.LINE_AA)
            if current_points:
                points = (np.asarray(current_points, dtype=np.float32) *
                          np.asarray([width, height], dtype=np.float32) * scale).astype(np.int32)
                cv2.polylines(display, [points], False, (0, 255, 255), 3, cv2.LINE_AA)
                for point in points:
                    cv2.circle(display, tuple(point), 5, (0, 165, 255), -1)
            panel = np.zeros((86, display_width, 3), dtype=np.uint8)
            cv2.putText(panel, "A: add | M: finish | E: edit vertex | Right-click: del polygon | D: del image | W: save | B: back | S: next | Q: quit",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, f"{current_split} {sample_index + 1}/{len(samples)} | labels: {len(polygons)} | add: {'ON' if adding else 'OFF'} | edit: {'ON' if edit_mode else 'OFF'}",
                        (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            canvas = np.vstack((display, panel))
            cv2.imshow(window_name, canvas)

            def mouse_callback(event, x, y, flags, parameter):
                nonlocal dirty, selected_vertex, dragging
                if edit_mode:
                    point = np.asarray([x / scale, y / scale], dtype=np.float32)
                    if event == cv2.EVENT_LBUTTONDOWN:
                        if y >= display_height:
                            return
                        nearest = None
                        nearest_distance = 20.0 / max(scale, 0.01)
                        for polygon_index, polygon in enumerate(polygons):
                            for vertex_index, vertex in enumerate(polygon * np.asarray([width, height], dtype=np.float32)):
                                distance = float(np.linalg.norm(vertex - point))
                                if distance < nearest_distance:
                                    nearest = (polygon_index, vertex_index)
                                    nearest_distance = distance
                        selected_vertex = nearest
                        dragging = nearest is not None
                    elif event == cv2.EVENT_MOUSEMOVE and dragging and selected_vertex is not None:
                        polygon_index, vertex_index = selected_vertex
                        polygons[polygon_index][vertex_index] = (
                            min(1.0, max(0.0, point[0] / width)),
                            min(1.0, max(0.0, point[1] / height)),
                        )
                        dirty = True
                    elif event == cv2.EVENT_LBUTTONUP:
                        dragging = False
                    return
                if adding and event == cv2.EVENT_LBUTTONDOWN and y < display_height:
                    current_points.append((x / scale / width, y / scale / height))
                elif event == cv2.EVENT_RBUTTONDOWN and y < display_height:
                    norm_point = (x / scale / width, y / scale / height)
                    for polygon_index in range(len(polygons) - 1, -1, -1):
                        if cv2.pointPolygonTest(polygons[polygon_index], norm_point, False) >= 0:
                            del polygons[polygon_index]
                            dirty = True
                            break
            cv2.setMouseCallback(window_name, mouse_callback)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("a"), ord("A")):
                adding = True
                edit_mode = False
            elif key in (ord("e"), ord("E")):
                edit_mode = not edit_mode
                adding = False
                selected_vertex = None
                dragging = False
            elif key in (ord("m"), ord("M")) and len(current_points) >= 3:
                polygons.append(np.asarray(current_points, dtype=np.float32))
                current_points = []
                adding = False
                dirty = True
            elif key in (ord("d"), ord("D")):
                os.remove(image_path)
                if os.path.exists(label_path):
                    os.remove(label_path)
                print(f"[標註檢查] 已刪除圖片與標註: {os.path.basename(image_path)}")
                del samples[sample_index]
                dirty = False
                break
            elif key in (ord("w"), ord("W")):
                with open(label_path, "w", encoding="utf-8") as label_file:
                    for polygon in polygons:
                        label_file.write("0 " + " ".join(f"{v:.6f}" for v in polygon.flatten()) + "\n")
                dirty = False
                print(f"[標註檢查] 已保存: {os.path.basename(image_path)}")
            elif key in (ord("s"), ord("S")):
                if dirty:
                    print(f"[標註檢查] 尚未保存，跳過: {os.path.basename(image_path)}")
                sample_index += 1
                break
            elif key in (ord("b"), ord("B")):
                if dirty:
                    print(f"[標註檢查] 尚未保存，返回上一張: {os.path.basename(image_path)}")
                sample_index = max(0, sample_index - 1)
                break
            elif key in (ord("q"), ord("Q"), 27):
                cv2.destroyWindow(window_name)
                return True
    cv2.destroyWindow(window_name)
    return True


def review_track_training_labels():
    """逐張檢查跑道訓練圖片，允許保留或刪除圖片與標註。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_root = os.path.join(base_dir, "track_training")
    image_dir = os.path.join(training_root, "images", "train")
    label_dir = os.path.join(training_root, "labels", "train")
    if not os.path.isdir(image_dir):
        print("[跑道資料檢查] 找不到 images/train 資料夾。")
        return False

    image_names = sorted(
        name for name in os.listdir(image_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    samples = [
        name for name in image_names
        if os.path.exists(os.path.join(label_dir, os.path.splitext(name)[0] + ".txt"))
    ]
    if not samples:
        print("[跑道資料檢查] 找不到有對應標註的圖片。")
        return False

    window_name = "Track Label Review - S: keep | D: delete | Q: quit"
    activate_english_keyboard_layout()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    apply_english_keyboard_layout_to_window(window_name)
    maximize_cv_window(window_name)
    sample_index = 0
    while sample_index < len(samples):
        image_name = samples[sample_index]
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + ".txt")
        frame = cv2.imread(image_path)
        if frame is None:
            sample_index += 1
            continue
        height, width = frame.shape[:2]
        labels = []
        with open(label_path, "r", encoding="utf-8") as label_file:
            for line in label_file:
                values = line.split()
                if len(values) >= 7 and (len(values) - 1) % 2 == 0:
                    class_id = int(float(values[0]))
                    polygon = np.asarray([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
                    labels.append((class_id, polygon))

        screen_width, screen_height = screen_size()
        display_width = min(width, max(640, screen_width - 40))
        scale = display_width / max(1, width)
        display_height = max(1, int(height * scale))
        display = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
        for class_id, polygon in labels:
            points = (polygon * np.asarray([width, height], dtype=np.float32) * scale).astype(np.int32)
            color = (0, 0, 255) if class_id == 0 else (0, 255, 0)
            name = "WALL" if class_id == 0 else "CLIP"
            cv2.polylines(display, [points], True, color, 4, cv2.LINE_AA)
            center = tuple(np.mean(points, axis=0).astype(int))
            cv2.putText(display, name, center, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        panel_height = 76
        panel = np.zeros((panel_height, display_width, 3), dtype=np.uint8)
        cv2.putText(panel, "S: keep / next | D: delete image + label | Q / ESC: quit",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"{sample_index + 1}/{len(samples)}  {image_name}  labels: {len(labels)}",
                    (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, np.vstack((display, panel)))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("d"), ord("D")):
            os.remove(image_path)
            os.remove(label_path)
            print(f"[跑道資料檢查] 已刪除: {image_name}")
            del samples[sample_index]
        elif key in (ord("s"), ord("S")):
            sample_index += 1
        elif key in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyWindow(window_name)
    print(f"[跑道資料檢查] 完成，目前保留 {len(samples)} 張。")
    return True


def _train_track_model(training_root, track_model_path, epochs=20):
    """只使用既有 track_training 資料訓練跑道 segmentation 模型。"""
    image_dir = os.path.join(training_root, "images", "train")
    if not os.path.isdir(image_dir):
        print(f"[跑道訓練] 找不到訓練圖片資料夾: {image_dir}")
        return False
    sample_count = len([name for name in os.listdir(image_dir)
                        if name.lower().endswith((".jpg", ".jpeg", ".png"))])
    if sample_count < 3:
        print(f"[跑道訓練] 需要至少 3 張標註圖片，目前只有 {sample_count} 張。")
        return False

    os.makedirs(training_root, exist_ok=True)
    yaml_path = os.path.join(training_root, "track_dataset.yaml")
    yaml_content = (
        f"path: {training_root.replace(os.sep, '/')}\n"
        "train: images/train\n"
        "val: images/train\n"
        "names:\n"
        "  0: Wall\n"
        "  1: Clip Zone\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as yaml_file:
        yaml_file.write(yaml_content)

    base_model = track_model_path if os.path.exists(track_model_path) else "yolov8n-seg.pt"
    print(f"[跑道訓練] 使用 {sample_count} 個既有樣本訓練 {epochs} epochs...")
    model = YOLO(base_model)
    model.train(data=yaml_path, epochs=epochs, imgsz=640, batch=16, workers=0,
                project=training_root, name="runs", exist_ok=True, verbose=False)
    best_model = os.path.join(training_root, "runs", "weights", "best.pt")
    if os.path.exists(best_model):
        shutil.copy2(best_model, track_model_path)
        print(f"[跑道訓練] 完成：{track_model_path}")
        return True
    print(f"[跑道訓練] 找不到訓練輸出的 best.pt: {best_model}")
    return False


def _train_vehicle_model(dataset_dir, epochs):
    image_dir = os.path.join(dataset_dir, "images", "train")
    image_count = len([name for name in os.listdir(image_dir) if name.lower().endswith(".jpg")])
    if image_count < 3:
        print(f"[車輛訓練] 需要至少 3 張標註圖片，目前只有 {image_count} 張。")
        return False

    # 建立獨立驗證集，避免用訓練圖片驗證造成過度樂觀的指標。
    val_image_dir = os.path.join(dataset_dir, "images", "val")
    val_label_dir = os.path.join(dataset_dir, "labels", "val")
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    existing_val_images = [name for name in os.listdir(val_image_dir)
                           if name.lower().endswith(".jpg")]
    if not existing_val_images and image_count >= 5:
        train_images = sorted(name for name in os.listdir(image_dir)
                              if name.lower().endswith(".jpg"))
        val_count = max(1, round(len(train_images) * 0.2))
        selected_indices = np.linspace(0, len(train_images) - 1, val_count, dtype=int)
        for index in sorted(set(selected_indices), reverse=True):
            image_name = train_images[index]
            label_name = os.path.splitext(image_name)[0] + ".txt"
            source_label = os.path.join(dataset_dir, "labels", "train", label_name)
            if not os.path.exists(source_label):
                continue
            shutil.move(os.path.join(image_dir, image_name), os.path.join(val_image_dir, image_name))
            shutil.move(source_label, os.path.join(val_label_dir, label_name))
        image_count = len([name for name in os.listdir(image_dir)
                           if name.lower().endswith(".jpg")])
        print(f"[車輛訓練] 已建立獨立驗證集：{val_count} 張，訓練集剩餘 {image_count} 張。")

    yaml_path = os.path.join(dataset_dir, "vehicle_dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as yaml_file:
        yaml_file.write(f"path: {dataset_dir.replace(os.sep, '/')}\ntrain: images/train\nval: images/val\nnames:\n  0: RC_Car\n")

    print(f"[車輛訓練] 使用 {image_count} 張圖片訓練 {epochs} epochs...")
    model = YOLO("yolov8n-seg.pt")
    model.train(data=yaml_path, epochs=epochs, imgsz=640, batch=4, workers=0,
                patience=30, project=dataset_dir, name="runs_latest", exist_ok=True)
    best_model = os.path.join(dataset_dir, "runs_latest", "weights", "best.pt")
    output_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rc_car_model.pt")
    if os.path.exists(best_model):
        shutil.copy2(best_model, output_model)
        print(f"[車輛訓練] 完成：{output_model}")
        return True
    print("[車輛訓練] 找不到訓練輸出的 best.pt。")
    return False

class DriftJudgeSystem:
    def __init__(self, video_path, track_model_path="track_seg_model.pt", sample_frames=5,
                 track_interval=1, enable_learning=False, car_model_path="yolov8n-seg.pt",
                 vehicle_class=2, track_merge_iou=0.35,
                 track_merge_center_ratio=0.55, track_merge_min_distance=30.0,
                 track_merge_containment=0.65):
        self.video_path = video_path
        self.sample_frames = max(1, int(sample_frames))
        self.track_interval = max(1, int(track_interval))
        self.enable_learning = bool(enable_learning)
        self.track_merge_iou = float(track_merge_iou)
        self.track_merge_center_ratio = float(track_merge_center_ratio)
        self.track_merge_min_distance = float(track_merge_min_distance)
        self.track_merge_containment = float(track_merge_containment)

        # 自訂遙控車模型應固定使用 class 0；若不是，可能誤用 COCO 的 vehicle 類別 (2)
        self.vehicle_class = 0 if os.path.basename(car_model_path).lower() in {"rc_car_model.pt", "best.pt"} else vehicle_class
        if self.vehicle_class == 2 and os.path.basename(car_model_path).lower() in {"rc_car_model.pt", "best.pt"}:
            self.vehicle_class = 0

        if not os.path.isabs(track_model_path):
            track_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), track_model_path)
        self.track_model_path = track_model_path
        self.training_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track_training")
        self.zones_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track_zones.json")
        
        # 1. 載入車輛追蹤模型 (YOLOv8 Segmentation)
        self.car_model = YOLO(car_model_path)
        
        # 賽道資訊設定 [(x1,y1), (x2,y2)...]
        self.walls = []       # 護欄多邊形列表
        self.clip_zones = []  # 得分區多邊形列表
        self.track_polygon = []
        
        # 歷史軌跡與成績
        self.car_history = {}
        self.track_aliases = {}
        self.track_last_seen = {}
        self.raw_track_last_seen = {}
        self.next_track_id = 1
        self.car_scores = {}
        self.car_events = {}
        self.car_details = {}
        self.car_score_totals = {}
        self.car_score_samples = {}
        self.car_last_observations = {}
        self.car_best_observations = {}
        self.track_color_indices = {}
        self.car_dominant_colors = {}
        self.video_fps = 30.0
        self.frame_diagonal = 1.0
        self.display_options = {
            "walls": True,
            "clips": True,
            "boxes": True,
            "axis": True,
        }

    def _save_zones(self):
        data = {
            "video_path": os.path.abspath(self.video_path),
            "walls": self.walls,
            "clip_zones": self.clip_zones,
        }
        with open(self.zones_path, "w", encoding="utf-8") as zones_file:
            json.dump(data, zones_file, ensure_ascii=False, indent=2)
        print(f"[區域保存] 護欄 {len(self.walls)} 個，得分區 {len(self.clip_zones)} 個")

    def _load_zones(self):
        if not os.path.exists(self.zones_path):
            return False
        try:
            with open(self.zones_path, "r", encoding="utf-8") as zones_file:
                data = json.load(zones_file)
            saved_video_path = data.get("video_path")
            if (not saved_video_path or
                    os.path.normcase(os.path.abspath(saved_video_path)) !=
                    os.path.normcase(os.path.abspath(self.video_path))):
                print("[區域提示] 已保存的跑道區域屬於其他影片，不載入。")
                return False
            self.walls = data.get("walls", [])
            self.clip_zones = data.get("clip_zones", [])
            self._rebuild_track_polygon()
            loaded = bool(self.walls or self.clip_zones)
            if loaded:
                print(f"[區域載入] 護欄 {len(self.walls)} 個，得分區 {len(self.clip_zones)} 個")
            return loaded
        except (OSError, json.JSONDecodeError) as error:
            print(f"[區域警告] 無法讀取 {self.zones_path}: {error}")
            return False

    def _rebuild_track_polygon(self):
        """以所有護欄的外側點建立跑道可行區域近似多邊形。"""
        if len(self.walls) < 2:
            self.track_polygon = []
            return
        points = np.concatenate([np.asarray(wall, dtype=np.int32) for wall in self.walls], axis=0)
        hull = cv2.convexHull(points)
        self.track_polygon = hull.reshape(-1, 2).tolist()

    def _vehicle_is_on_track(self, center):
        if len(self.walls) < 2 or len(self.track_polygon) < 3:
            return True
        return cv2.pointPolygonTest(
            np.asarray(self.track_polygon, dtype=np.int32), center, False
        ) >= 0

    def _consolidate_detections(self, boxes, confidences, masks, track_ids):
        """合併高度重疊/互相包含的同車偵測，保留最高信心輪廓。"""
        if len(boxes) < 2:
            return list(range(len(boxes)))
        order = sorted(range(len(boxes)), key=lambda index: float(confidences[index]), reverse=True)
        kept = []
        for index in order:
            box = boxes[index]
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            duplicate = False
            for kept_index in kept:
                other = boxes[kept_index]
                other_area = max(0.0, other[2] - other[0]) * max(0.0, other[3] - other[1])
                intersection = (
                    max(0.0, min(box[2], other[2]) - max(box[0], other[0])) *
                    max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
                )
                union = area + other_area - intersection
                smaller = max(min(area, other_area), 1.0)
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                other_center = ((other[0] + other[2]) / 2, (other[1] + other[3]) / 2)
                center_distance = math.hypot(
                    center[0] - other_center[0], center[1] - other_center[1]
                )
                size_limit = max(
                    self.track_merge_min_distance,
                    min(box[2] - box[0], box[3] - box[1],
                        other[2] - other[0], other[3] - other[1]) * self.track_merge_center_ratio,
                )
                close_centers = center_distance <= size_limit
                if ((union > 0 and intersection / union >= self.track_merge_iou) or
                    intersection / smaller >= self.track_merge_containment or close_centers):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(index)
        return sorted(kept)

    def _stable_track_id(self, raw_track_id, center, frame_idx):
        """將追蹤器重新產生的 ID 合併回最近的既有軌跡。"""
        raw_track_id = int(raw_track_id)
        if raw_track_id in self.track_aliases:
            if self.raw_track_last_seen.get(raw_track_id) == frame_idx - 1:
                self.raw_track_last_seen[raw_track_id] = frame_idx
                return self.track_aliases[raw_track_id]
            del self.track_aliases[raw_track_id]
        if raw_track_id in self.car_history and self.track_last_seen.get(raw_track_id) == frame_idx - 1:
            self.raw_track_last_seen[raw_track_id] = frame_idx
            return raw_track_id

        max_distance = max(80.0, self.frame_diagonal * 0.08)
        nearest_id = None
        nearest_distance = max_distance
        for existing_id, history in self.car_history.items():
            last_seen = self.track_last_seen.get(existing_id, -1)
            if not history or frame_idx - last_seen > max(10, self.track_interval * 5):
                continue
            distance = math.hypot(
                center[0] - history[-1][0][0],
                center[1] - history[-1][0][1],
            )
            if distance < nearest_distance:
                nearest_id = existing_id
                nearest_distance = distance
        if nearest_id is not None:
            self.track_aliases[raw_track_id] = nearest_id
            self.raw_track_last_seen[raw_track_id] = frame_idx
            return nearest_id

        stable_id = raw_track_id
        if stable_id in self.car_history:
            stable_id = max(self.next_track_id, max(self.car_history, default=0) + 1)
            while stable_id in self.car_history:
                stable_id += 1
        self.next_track_id = max(self.next_track_id, stable_id + 1)
        self.raw_track_last_seen[raw_track_id] = frame_idx
        return stable_id

    def _reset_analysis_state(self):
        """清除上一輪分析結果，供重播使用。"""
        self.car_history.clear()
        self.track_aliases.clear()
        self.track_last_seen.clear()
        self.raw_track_last_seen.clear()
        self.next_track_id = 1
        self.car_scores.clear()
        self.car_events.clear()
        self.car_details.clear()
        self.car_score_totals.clear()
        self.car_score_samples.clear()
        self.car_last_observations.clear()
        self.car_best_observations.clear()
        self.track_color_indices.clear()
        self.car_dominant_colors.clear()
        self.car_last_observations.clear()
        self.car_best_observations.clear()

    def _training_paths(self):
        image_dir = os.path.join(self.training_root, "images", "train")
        label_dir = os.path.join(self.training_root, "labels", "train")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        return image_dir, label_dir

    def _save_manual_training_sample(self, frame):
        """保存手繪畫面與 YOLO segmentation 標籤，供下次自動辨識使用。"""
        if not self.walls and not self.clip_zones:
            return False

        image_dir, label_dir = self._training_paths()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(image_dir, f"track_{stamp}.jpg")
        label_path = os.path.join(label_dir, f"track_{stamp}.txt")
        height, width = frame.shape[:2]
        labels = []

        for class_id, zones in ((0, self.walls), (1, self.clip_zones)):
            for zone in zones:
                polygon = np.asarray(zone, dtype=np.float32)
                if len(polygon) < 3:
                    continue
                normalized = polygon.copy()
                normalized[:, 0] /= width
                normalized[:, 1] /= height
                coordinates = " ".join(f"{value:.6f}" for value in normalized.flatten())
                labels.append(f"{class_id} {coordinates}")

        if not labels or not cv2.imwrite(image_path, frame):
            return False
        with open(label_path, "w", encoding="utf-8") as label_file:
            label_file.write("\n".join(labels) + "\n")
        print(f"[學習資料] 已保存手繪樣本: {os.path.basename(image_path)}")
        return True

    def _training_sample_count(self):
        image_dir = os.path.join(self.training_root, "images", "train")
        if not os.path.isdir(image_dir):
            return 0
        return len([name for name in os.listdir(image_dir) if name.lower().endswith((".jpg", ".jpeg", ".png"))])

    def _train_from_manual_samples(self):
        """累積足夠手繪樣本後，自動微調跑道模型。"""
        _train_track_model(self.training_root, self.track_model_path, epochs=20)

    @staticmethod
    def _prepare_display_window(window_name, frame):
        """建立可調整大小的視窗，避免高解析度影片只顯示左上角。"""
        height, width = frame.shape[:2]
        screen_width, screen_height = screen_size()
        max_width = max(640, screen_width - 20)
        max_height = max(480, screen_height - 80)
        scale = min(max_width / width, max_height / height)
        display_width = max(1, int(width * scale))
        display_height = max(1, int(height * scale))

        activate_english_keyboard_layout()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        apply_english_keyboard_layout_to_window(window_name)
        maximize_cv_window(window_name)
        return display_width, display_height, scale

    def _apply_nearest_track_sample(self, frame):
        """以目前影片第一幀找最接近的手繪樣張，套用其跑道 polygon。"""
        image_dir = os.path.join(self.training_root, "images", "train")
        label_dir = os.path.join(self.training_root, "labels", "train")
        if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
            return False

        frame_height, frame_width = frame.shape[:2]
        target_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 96),
                                 interpolation=cv2.INTER_AREA)
        target_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        target_hist = cv2.normalize(
            cv2.calcHist([target_hsv], [0, 1], None, [24, 16], [0, 180, 0, 256]),
            None).flatten()
        best_score = None
        best_paths = None

        for image_name in os.listdir(image_dir):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + ".txt")
            if not os.path.exists(label_path):
                continue
            sample = cv2.imread(image_path)
            if sample is None:
                continue
            sample_gray = cv2.resize(cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY), (96, 96),
                                     interpolation=cv2.INTER_AREA)
            sample_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
            sample_hist = cv2.normalize(
                cv2.calcHist([sample_hsv], [0, 1], None, [24, 16], [0, 180, 0, 256]),
                None).flatten()
            gray_error = float(cv2.absdiff(target_gray, sample_gray).mean()) / 255.0
            hist_error = 1.0 - float(cv2.compareHist(target_hist, sample_hist, cv2.HISTCMP_CORREL))
            aspect_error = abs((frame_width / max(1, frame_height)) -
                               (sample.shape[1] / max(1, sample.shape[0])))
            score = gray_error * 0.65 + hist_error * 0.30 + aspect_error * 0.05
            if best_score is None or score < best_score:
                best_score = score
                best_paths = (image_path, label_path)

        if best_paths is None:
            return False

        self.walls.clear()
        self.clip_zones.clear()
        image_path, label_path = best_paths
        with open(label_path, "r", encoding="utf-8") as label_file:
            for line in label_file:
                values = line.split()
                if len(values) < 7 or (len(values) - 1) % 2 != 0:
                    continue
                class_id = int(float(values[0]))
                normalized = np.asarray([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
                polygon = (normalized * np.asarray([frame_width, frame_height], dtype=np.float32)).astype(np.int32).tolist()
                if class_id == 0:
                    self.walls.append(polygon)
                elif class_id == 1:
                    self.clip_zones.append(polygon)
        self._rebuild_track_polygon()
        print(f"[樣張套用] 使用最相似樣張: {os.path.basename(image_path)} | score={best_score:.4f} | "
              f"護欄 {len(self.walls)} 個，得分區 {len(self.clip_zones)} 個")
        return bool(self.walls or self.clip_zones)

    def auto_detect_track(self, frame):
        """先套用最相似手繪樣張，找不到樣張時才使用跑道模型。"""
        self.walls.clear()
        self.clip_zones.clear()
        self.track_polygon = []
        if self._apply_nearest_track_sample(frame):
            return True
        if not os.path.exists(self.track_model_path):
            print(f"[提示] 未找到賽道模型 {self.track_model_path}，直接進入方案 B 手動標註。")
            return False

        print("\n[方案 A] 正在進行賽道與得分區自動辨識...")
        track_model = YOLO(self.track_model_path)
        
        results = track_model(frame, verbose=False)
        
        if results[0].masks is not None:
            classes = results[0].boxes.cls.cpu().numpy()
            masks = results[0].masks.xy

            self.walls.clear()
            self.clip_zones.clear()

            for i, cls in enumerate(classes):
                polygon = masks[i].astype(np.int32).tolist()
                if int(cls) == 0:    # Class 0: Wall
                    self.walls.append(polygon)
                elif int(cls) == 1:  # Class 1: Clip Zone
                    self.clip_zones.append(polygon)
            self._rebuild_track_polygon()
            
            print(f"[方案 A 成功] 自動偵測到 {len(self.walls)} 個護欄區域, {len(self.clip_zones)} 個得分區。")
            if self.walls or self.clip_zones:
                return True
            print("[方案 A 警告] 找到物件但沒有 Wall/Clip Zone 類別，進入方案 B 手動標註。")
            return False
        else:
            print("[方案 A 警告] 未能在畫面中辨識出賽道區域，進入方案 B 手動標註。")
            return False

    def _draw_track_zones(self, frame, show_labels=True):
        """以半透明填色和粗邊框顯示護欄與得分區。"""
        overlay = frame.copy()
        if self.display_options["walls"]:
            for index, wall in enumerate(self.walls, start=1):
                polygon = np.array(wall, dtype=np.int32)
                cv2.fillPoly(overlay, [polygon], (0, 0, 180))
                cv2.polylines(frame, [polygon], True, (0, 0, 255), 6, cv2.LINE_AA)
                center = np.mean(polygon, axis=0).astype(int)
                if show_labels:
                    cv2.putText(frame, f"WALL {index}", tuple(center),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
        if self.display_options["clips"]:
            for index, clip in enumerate(self.clip_zones, start=1):
                polygon = np.array(clip, dtype=np.int32)
                cv2.fillPoly(overlay, [polygon], (0, 180, 0))
                cv2.polylines(frame, [polygon], True, (0, 255, 0), 6, cv2.LINE_AA)
                center = np.mean(polygon, axis=0).astype(int)
                if show_labels:
                    cv2.putText(frame, f"CLIP ZONE {index}", tuple(center),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
        return frame

    def setup_zones_interactive(self):
        """展示方案 A 結果，並提供方案 B 手動修訂/切換介面"""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("無法讀取影片預覽幀，請確認影片檔案格式是否正常！")
            return False

        # 執行方案 A 自動辨識
        auto_success = self.auto_detect_track(frame)

        print("\n" + "="*50)
        print("【賽道標註與確認介面】")
        print(" - 按 [ENTER]  : 滿意辨識結果，接受並開始分析")
        print(" - 按 [W]      : 儲存目前護欄")
        print(" - 按 [Z]      : 儲存目前得分區")
        print(" - 按 [C]      : 取消目前未完成 polygon")
        print(" - 按 [R]      : 清除全部護欄與得分區")
        print(" - [方案 B 手動操作說明]:")
        print("   1. 滑鼠左鍵點擊畫面繪製多邊形頂點")
        print("   2. 按 [W] 儲存護欄，[Z] 儲存得分區")
        print("   3. 按 [1]/[2] 切換目前繪製類別")
        print("   4. 按 [ENTER] 儲存區域並接受，按 [Q]/[ESC] 取消")
        print("="*50 + "\n")

        current_pts = []
        mode = "WALL"  # WALL or CLIP
        is_manual_mode = not auto_success

        setup_window = "Track Setup - Plan A & B Confirm"
        display_width, display_height, display_scale = self._prepare_display_window(setup_window, frame)
        accepted = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal is_manual_mode
            if event == cv2.EVENT_LBUTTONDOWN:
                is_manual_mode = True
                original_x = min(frame.shape[1] - 1, max(0, int(x / display_scale)))
                original_y = min(frame.shape[0] - 1, max(0, int(y / display_scale)))
                current_pts.append((original_x, original_y))

        cv2.setMouseCallback(setup_window, mouse_callback)

        while True:
            display = frame.copy()

            # 繪製護欄 (紅色) 與 得分區 (綠色)
            for w in self.walls:
                cv2.polylines(display, [np.array(w, dtype=np.int32)], True, (0, 0, 255), 2)
                cv2.fillPoly(display, [np.array(w, dtype=np.int32)], (0, 0, 100))
                wall_center = np.mean(np.array(w, dtype=np.float32), axis=0).astype(int)
                cv2.putText(display, "WALL / 護欄", tuple(wall_center),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            for c in self.clip_zones:
                cv2.polylines(display, [np.array(c, dtype=np.int32)], True, (0, 255, 0), 2)
                cv2.fillPoly(display, [np.array(c, dtype=np.int32)], (0, 100, 0))
                clip_center = np.mean(np.array(c, dtype=np.float32), axis=0).astype(int)
                cv2.putText(display, "CLIP ZONE / 得分區", tuple(clip_center),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # 繪製手動標註中之點位 (黃色)
            if len(current_pts) > 0:
                cv2.polylines(display, [np.array(current_pts)], False, (0, 255, 255), 2)
                for pt in current_pts:
                    cv2.circle(display, pt, 4, (0, 255, 255), -1)

            display_frame = cv2.resize(display, (display_width, display_height), interpolation=cv2.INTER_AREA)
            status_source = "AUTO DETECTED" if auto_success and not is_manual_mode else "MANUAL DRAWING"
            mode_label = "DRAWING WALL / 護欄" if mode == "WALL" else "DRAWING CLIP ZONE / 得分區"
            status_text = f"{status_source} | {mode_label} | Walls: {len(self.walls)} | Clips: {len(self.clip_zones)}"
            status_color = (0, 255, 0) if auto_success and not is_manual_mode else (0, 255, 255)
            cv2.putText(display_frame, status_text, (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

            if auto_success:
                auto_message = "AUTO RESULT: review zones, W/Z save, ENTER accept, Q cancel"
            else:
                auto_message = "MANUAL: W wall | Z clip | C cancel current | R clear all"
            cv2.putText(display_frame, auto_message, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

            instructions = [
                "LEFT CLICK: add point   W: save Wall   Z: save Clip   U: undo",
                "1/2: type   C: cancel current   R: clear all   ENTER: accept   Q: cancel"
            ]
            panel_top = max(0, display_height - 88)
            cv2.rectangle(display_frame, (10, panel_top), (display_width - 10, display_height - 10), (0, 0, 0), -1)
            for index, instruction in enumerate(instructions):
                cv2.putText(display_frame, instruction, (22, panel_top + 30 + index * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(setup_window, display_frame)

            key = cv2.waitKey(20) & 0xFF
            
            if key == 13:  # ENTER 鍵
                accepted = True
                break
            elif key in (ord('q'), ord('Q'), 27):
                accepted = False
                break
            elif key in (ord('c'), ord('C')):
                current_pts.clear()
                print("[方案 B] 已取消目前未完成的 polygon。")
            elif key in (ord('r'), ord('R')):
                self.walls.clear()
                self.clip_zones.clear()
                current_pts.clear()
                is_manual_mode = True
                print("[系統] 已清空標註，啟動 [方案 B] 手動繪製模式。")
            elif key in (ord('w'), ord('W'), ord('z'), ord('Z'), ord('n'), ord('N')):
                if len(current_pts) >= 3:
                    save_as_wall = key in (ord('w'), ord('W')) or (
                        key in (ord('n'), ord('N')) and mode == "WALL"
                    )
                    if save_as_wall:
                        self.walls.append(current_pts.copy())
                        saved_mode = "WALL"
                    else:
                        self.clip_zones.append(current_pts.copy())
                        saved_mode = "CLIP"
                    current_pts.clear()
                    self._rebuild_track_polygon()
                    print(f"[方案 B] 已新增 {saved_mode} 區域。")
            elif key in (ord('d'), ord('D')):
                mode = "CLIP" if mode == "WALL" else "WALL"
                current_pts.clear()
                print(f"[方案 B] 當前標註類別切換為: {mode}")
            elif key == ord('1'):
                mode = "WALL"
                current_pts.clear()
                print("[方案 B] 當前標註類別: WALL / 護欄")
            elif key == ord('2'):
                mode = "CLIP"
                current_pts.clear()
                print("[方案 B] 當前標註類別: CLIP / 得分區")
            elif key in (ord('u'), ord('U')) and current_pts:
                current_pts.pop()
                print("[方案 B] 已撤銷上一個點。")

        cv2.destroyAllWindows()
        if accepted and is_manual_mode:
            self._rebuild_track_polygon()
            self._save_zones()
            if self._save_manual_training_sample(frame):
                if self.enable_learning:
                    self._train_from_manual_samples()
                else:
                    print("[自動學習] 已關閉，本次只保存手繪資料。下次使用 --learn 才會訓練模型。")
        elif accepted:
            self._save_zones()
        return accepted

    def calculate_metrics(self, history, bbox=None):
        """依固定權重計算單次取樣分數，總分上限為 100。"""
        if len(history) < 2:
            return {
                "speed": 0, "drift_angle": 0, "is_spin": False, "hit_wall": False,
                "score_delta": 0, "total_score": 0, "line_score": 0,
                "angle_score": 0, "speed_score": 0, "speed_stability_score": 0,
                "angle_stability_score": 0, "wall_distance": None,
                "speed_mean": 0, "speed_std": 0, "angle_std": 0,
                "route_ratio": 0,
            }

        p_curr, a_curr, f_curr = history[-1]
        p_prev, a_prev, f_prev = history[-2]
        
        # 1. 速度計算
        dt = f_curr - f_prev
        dist = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        speed = dist * self.video_fps / dt if dt > 0 else 0

        # 2. 夾角計算
        move_vec = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        move_angle = math.degrees(math.atan2(move_vec[1], move_vec[0])) % 360
        drift_angle = abs(a_curr - move_angle)
        if drift_angle > 180:
            drift_angle = 360 - drift_angle

        # 3. Spin 判斷
        angle_diff = abs(a_curr - a_prev)
        if angle_diff > 180: angle_diff = 360 - angle_diff
        is_spin = angle_diff > 100 and speed / max(self.frame_diagonal, 1) < 0.15

        # 4. 撞牆與離牆距離
        hit_wall = False
        wall_distance = None
        if self.walls:
            wall_distances = []
            check_points = [p_curr]
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                check_points.extend([(x1, y1), (x2, y1), (x1, y2), (x2, y2)])
            for wall in self.walls:
                polygon = np.array(wall, dtype=np.int32)
                wall_distances.append(abs(cv2.pointPolygonTest(polygon, p_curr, True)))
                if any(cv2.pointPolygonTest(polygon, point, False) >= 0 for point in check_points):
                    hit_wall = True
            wall_distance = min(wall_distances)

        # 最近一段軌跡用於計算穩定度，避免只用兩幀造成分數跳動。
        recent_history = history[-min(len(history), 30):]
        speeds = []
        angles = []
        for previous_item, current_item in zip(recent_history, recent_history[1:]):
            previous_point, previous_angle, previous_frame = previous_item
            current_point, current_angle, current_frame = current_item
            frame_delta = current_frame - previous_frame
            if frame_delta <= 0:
                continue
            interval_speed = math.hypot(
                current_point[0] - previous_point[0],
                current_point[1] - previous_point[1],
            ) * self.video_fps / frame_delta
            movement = math.degrees(math.atan2(
                current_point[1] - previous_point[1],
                current_point[0] - previous_point[0],
            )) % 360
            interval_angle = abs(current_angle - movement)
            if interval_angle > 180:
                interval_angle = 360 - interval_angle
            speeds.append(interval_speed)
            angles.append(interval_angle)

        speed_mean = float(np.mean(speeds)) if speeds else speed
        speed_std = float(np.std(speeds)) if speeds else 0.0
        angle_std = float(np.std(angles)) if angles else 0.0

        # 固定權重：路線 40、角度 20、速度 10、穩定度 30。
        # 使用容許範圍校準畫面像素，避免正常影片因解析度或偵測微抖被壓到低分。
        normalized_wall_distance = wall_distance / max(self.frame_diagonal, 1) if wall_distance is not None else 0.0
        # 以較寬的安全距離區間評估路線，避免牆面標註厚度或解析度使分數變成 0。
        route_quality = min(1.0, normalized_wall_distance / 0.02, (0.20 - normalized_wall_distance) / 0.12)
        line_score = max(0.0, min(40.0, route_quality * 40.0)) if wall_distance is not None else 0.0
        angle_quality = 1.0 - abs(drift_angle - 45.0) / 45.0
        angle_score = max(0.0, min(20.0, angle_quality * 20.0))
        speed_score = min(10.0, speed / max(self.frame_diagonal * 0.025, 1) * 10.0)
        speed_stability_score = max(0.0, min(10.0, 10.0 * (1.0 - speed_std / max(speed_mean * 0.50, 1.0))))
        angle_stability_score = max(0.0, min(20.0, 20.0 * (1.0 - angle_std / 35.0)))
        if hit_wall:
            line_score = 0.0
        if is_spin:
            angle_stability_score = 0.0
        total_score = min(100.0, max(0.0, line_score + angle_score + speed_score +
                                     speed_stability_score + angle_stability_score))

        return {
            "speed": round(speed, 2),
            "drift_angle": round(drift_angle, 1),
            "is_spin": is_spin,
            "hit_wall": hit_wall,
            "score_delta": round(total_score, 1)
            ,"total_score": round(total_score, 1)
            ,"wall_distance": round(wall_distance, 1) if wall_distance is not None else None
            ,"line_score": round(line_score, 1)
            ,"speed_score": round(speed_score, 1)
            ,"angle_score": round(angle_score, 1)
            ,"speed_stability_score": round(speed_stability_score, 1)
            ,"angle_stability_score": round(angle_stability_score, 1)
            ,"speed_mean": round(speed_mean, 2)
            ,"speed_std": round(speed_std, 2)
            ,"angle_std": round(angle_std, 2)
            ,"route_ratio": round(normalized_wall_distance, 4)
        }

    def _save_score_results(self):
        """保存本次影片的分數與分項結果，供後續調整計分公式。"""
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_results")
        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(output_dir, f"score_{stamp}.json")
        result = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "video_path": os.path.abspath(self.video_path),
            "video_fps": self.video_fps,
            "walls": self.walls,
            "clip_zones": self.clip_zones,
            "cars": {},
        }
        for track_id, score in self.car_scores.items():
            result["cars"][str(track_id)] = {
                "score": round(float(score), 2),
                "events": self.car_events.get(track_id, {}),
                "details": self.car_details.get(track_id, {}),
                "samples": self.car_score_samples.get(track_id, 0),
                "last_observation": self.car_last_observations.get(track_id, {}),
            }
        with open(result_path, "w", encoding="utf-8") as result_file:
            json.dump(result, result_file, ensure_ascii=False, indent=2)
        print(f"[分數保存] 已保存: {result_path}")
        return result_path

    @staticmethod
    def _smooth_trajectory(points, window=7):
        """用滑動平均降低偵測抖動，保留軌跡整體走向。"""
        points = np.asarray(points, dtype=np.float32)
        if len(points) < 3:
            return points.astype(np.int32)
        half_window = max(1, window // 2)
        smoothed = []
        for index in range(len(points)):
            start = max(0, index - half_window)
            end = min(len(points), index + half_window + 1)
            smoothed.append(np.mean(points[start:end], axis=0))
        return np.asarray(smoothed, dtype=np.int32)

    @staticmethod
    def _rear_point(center, angle_degrees, half_length):
        """依車輛朝向與長度，將中心點換算為車尾位置（角度線非箭頭那端）。"""
        radians_value = math.radians(float(angle_degrees))
        offset_x = math.cos(radians_value) * half_length
        offset_y = math.sin(radians_value) * half_length
        return (center[0] - offset_x, center[1] - offset_y)

    def _rear_trajectory_points(self, history, half_length):
        """將歷史中心點依各幀角度換算為車尾軌跡點，供軌跡線與角度線共用同一基準。"""
        return [self._rear_point(center, angle, half_length) for center, angle, _frame_idx in history]

    @staticmethod
    def _extract_dominant_color(frame, polygon, bbox):
        """取樣車輛遮罩內主要（大面積）顏色，用於軌跡線與角度線上色。"""
        if frame is None:
            return None
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        if polygon is not None and len(polygon) >= 3:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        elif bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = [int(value) for value in bbox]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        else:
            return None
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
        pixels = frame[mask > 0].astype(np.float32)
        if pixels.shape[0] < 20:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        cluster_count = 3 if pixels.shape[0] >= 3 else 1
        try:
            _compactness, labels, centers = cv2.kmeans(
                pixels, cluster_count, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(labels.flatten(), minlength=cluster_count)
            dominant_color = centers[int(np.argmax(counts))]
        except cv2.error:
            dominant_color = pixels.mean(axis=0)
        return tuple(int(value) for value in dominant_color)

    def _track_annotation_color(self, track_id):
        """優先使用車輛實際主色，尚未取樣時以出場順序配色作為備援。"""
        track_id = int(track_id)
        dominant_color = self.car_dominant_colors.get(track_id)
        if dominant_color is not None:
            return dominant_color
        palette = (
            (0, 165, 255),    # orange
            (0, 220, 80),     # green
            (255, 120, 0),    # blue
            (255, 0, 220),    # magenta
            (0, 220, 220),    # yellow
            (220, 80, 255),   # pink
        )
        if track_id not in self.track_color_indices:
            self.track_color_indices[track_id] = len(self.track_color_indices) % len(palette)
        return palette[self.track_color_indices[track_id]]

    @staticmethod
    def _continuous_axis_angle(previous_angle, current_angle):
        """選擇與上一幀最接近的 180 度等價角，避免車身方向線反轉。"""
        if previous_angle is None:
            return float(current_angle)
        current_angle = float(current_angle)
        previous_angle = float(previous_angle)
        current_angle += 180.0 * round((previous_angle - current_angle) / 180.0)
        return current_angle

    def process_video(self):
        """執行比賽動態追蹤與判分"""
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0
        self.display_options.update({
            "walls": True,
            "clips": True,
            "boxes": True,
            "axis": True,
        })
        self.car_details.clear()
        self.car_score_totals.clear()
        self.car_score_samples.clear()

        print("\n開始分析影片，按 [q] 可隨時停止...")
        display_window = "RC Drift Analyzer"
        window_initialized = False
        display_width = display_height = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30
        self.video_fps = fps
        track_interval = self.track_interval
        playback_start = time.perf_counter()
        last_frame = None
        last_original_frame = None
        last_display_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            last_frame = frame.copy()
            last_original_frame = frame.copy()

            if not window_initialized:
                display_width, display_height, _ = self._prepare_display_window(display_window, frame)
                self.frame_diagonal = math.hypot(frame.shape[1], frame.shape[0])
                window_initialized = True
            
            frame_idx += 1
            current_frame_metrics = {}
            
            # 限制推論頻率，並不額外等待每幀的影片時間，避免 CPU 推論拖慢播放。
            should_track = frame_idx % track_interval == 0
            results = None
            if should_track:
                tracking_classes = None if self.vehicle_class < 0 else [self.vehicle_class]
                results = self.car_model.track(frame, persist=True, classes=tracking_classes,
                                               verbose=False, imgsz=640)

            # 畫出最終賽道與得分區
            self._draw_track_zones(frame, show_labels=False)

            if results is not None and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                detected_ids = results[0].boxes.id
                if detected_ids is not None:
                    track_ids = detected_ids.int().cpu().numpy()
                else:
                    track_ids = np.arange(len(boxes))
                masks = results[0].masks.xy if results[0].masks is not None else []
                confidences = results[0].boxes.conf.cpu().numpy()
                kept_indices = self._consolidate_detections(boxes, confidences, masks, track_ids)

                for i in kept_indices:
                    track_id = track_ids[i]
                    box = boxes[i].astype(int)
                    if self.display_options["boxes"]:
                        box_color = (255, 255, 0) if detected_ids is not None else (0, 165, 255)
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), box_color, 5)
                    if i < len(masks):
                        polygon = masks[i].astype(np.int32)
                        if self.display_options["boxes"]:
                            cv2.polylines(frame, [polygon], True, (255, 255, 0), 2)
                        rect = cv2.minAreaRect(polygon)
                        center, size, angle = rect
                        center = (int(center[0]), int(center[1]))
                        axis_angle = angle if size[0] >= size[1] else angle + 90
                    else:
                        center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                        angle = 0
                        axis_angle = 0

                    if not self._vehicle_is_on_track(center):
                        continue
                    track_id = self._stable_track_id(track_id, center, frame_idx)
                    self.track_last_seen[track_id] = frame_idx
                    previous_observation = self.car_last_observations.get(track_id)
                    previous_angle = (previous_observation.get("angle")
                                      if previous_observation else None)
                    axis_angle = self._continuous_axis_angle(previous_angle, axis_angle)

                    self.car_last_observations[track_id] = {
                        "center": [int(center[0]), int(center[1])],
                        "angle": round(float(axis_angle), 1),
                        "confidence": round(float(confidences[i]), 4),
                        "bbox": [int(value) for value in box],
                        "polygon": masks[i].astype(int).tolist() if i < len(masks) else [],
                    }
                    current_observation = self.car_last_observations[track_id]
                    bbox_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
                    observation_quality = bbox_area * float(confidences[i])
                    best_observation = self.car_best_observations.get(track_id)
                    if best_observation is None or observation_quality > best_observation["quality"]:
                        self.car_best_observations[track_id] = {
                            "quality": observation_quality,
                            "bbox": current_observation["bbox"].copy(),
                            "polygon": current_observation["polygon"].copy(),
                        }
                        best_observation = self.car_best_observations[track_id]
                        dominant_color = self._extract_dominant_color(
                            last_original_frame, current_observation["polygon"], current_observation["bbox"])
                        if dominant_color is not None:
                            self.car_dominant_colors[track_id] = dominant_color

                    # 角度線長度直接取評分時套上的水藍色外框（最佳觀測）長度，不再每幀重新採樣
                    best_bbox = best_observation["bbox"]
                    axis_half_length = max(20, max(best_bbox[2] - best_bbox[0], best_bbox[3] - best_bbox[1])) / 2
                    axis_radians = math.radians(axis_angle)
                    axis_delta = (int(math.cos(axis_radians) * axis_half_length),
                                  int(math.sin(axis_radians) * axis_half_length))
                    axis_start = (center[0] - axis_delta[0], center[1] - axis_delta[1])
                    axis_end = (center[0] + axis_delta[0], center[1] + axis_delta[1])
                    if self.display_options["axis"]:
                        cv2.arrowedLine(frame, axis_start, axis_end, (0, 165, 255), 3, cv2.LINE_AA, tipLength=0.25)
                        cv2.circle(frame, center, 5, (0, 165, 255), -1)

                    if track_id not in self.car_history:
                        self.car_history[track_id] = []
                        self.car_scores[track_id] = 0
                        self.car_events[track_id] = {"crashes": 0, "spins": 0}
                        self.car_score_totals[track_id] = {
                            "line_score": 0.0,
                            "angle_score": 0.0,
                            "speed_score": 0.0,
                            "speed_stability_score": 0.0,
                            "angle_stability_score": 0.0,
                        }
                        self.car_score_samples[track_id] = 0

                    self.car_history[track_id].append((center, axis_angle, frame_idx))
                    current_metrics = self.calculate_metrics(self.car_history[track_id], box.tolist())
                    current_frame_metrics[track_id] = current_metrics.copy()

                    # 每 X 幀計算一次
                    if frame_idx % self.sample_frames == 0:
                        metrics = current_metrics
                        self.car_last_observations[track_id]["speed"] = metrics["speed"]
                        self.car_last_observations[track_id]["drift_angle"] = metrics["drift_angle"]
                        totals = self.car_score_totals[track_id]
                        for component in totals:
                            totals[component] += metrics[component]
                        self.car_score_samples[track_id] += 1
                        sample_count = self.car_score_samples[track_id]
                        self.car_scores[track_id] = min(100.0, max(0.0, sum(
                            totals[component] / sample_count for component in totals
                        )))
                        if metrics["hit_wall"]:
                            self.car_events[track_id]["crashes"] += 1
                        if metrics["is_spin"]:
                            self.car_events[track_id]["spins"] += 1
                        self.car_details[track_id] = {
                            key: value / sample_count if key in totals else value
                            for key, value in metrics.items()
                        }
                        
                        cv2.circle(frame, center, 4, (0, 0, 255), -1)

            display_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
            cv2.putText(display_frame, "Enter-離開", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            for row, track_id in enumerate(sorted(current_frame_metrics), start=1):
                metrics = current_frame_metrics[track_id]
                line = (f"car{row:02d}  速度 {metrics['speed']:.1f}  角度 {abs(round(metrics['drift_angle']))}°  "
                        f"路線 {metrics['line_score']:.1f}")
                cv2.putText(display_frame, line, (20, 38 + row * 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            last_display_frame = display_frame.copy()
            cv2.imshow(display_window, display_frame)
            target_time = playback_start + max(0, frame_idx - 1) / self.video_fps
            wait_ms = max(1, int((target_time - time.perf_counter()) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord('q'), 13):
                break
            elif key == ord('w'):
                self.display_options["walls"] = not self.display_options["walls"]
            elif key == ord('z'):
                self.display_options["clips"] = not self.display_options["clips"]
            elif key == ord('b'):
                self.display_options["boxes"] = not self.display_options["boxes"]
            elif key == ord('a'):
                self.display_options["axis"] = not self.display_options["axis"]
            elif key == ord('h'):
                show_all = not all(self.display_options.values())
                for option in self.display_options:
                    self.display_options[option] = show_all

        cap.release()
        if last_original_frame is not None or last_display_frame is not None:
            summary_frame = (last_original_frame.copy() if last_original_frame is not None
                             else last_display_frame.copy())
            summary_scale = display_width / max(1, summary_frame.shape[1])

            def summary_font(size):
                return max(0.3, size / max(summary_scale, 0.01))

            def summary_thickness(size):
                return max(1, int(round(size / max(summary_scale, 0.01))))

            def summary_ui_point(x, y):
                return (int(x / max(summary_scale, 0.01)),
                        int(y / max(summary_scale, 0.01)))

            def summary_ui_rect(x1, y1, x2, y2):
                return (*summary_ui_point(x1, y1), *summary_ui_point(x2, y2))

            for track_id, history in self.car_history.items():
                best_observation_for_length = self.car_best_observations.get(track_id)
                best_bbox_for_length = best_observation_for_length.get("bbox", []) if best_observation_for_length else []
                rear_half_length = (max(20, max(best_bbox_for_length[2] - best_bbox_for_length[0],
                                                 best_bbox_for_length[3] - best_bbox_for_length[1])) / 2
                                    if len(best_bbox_for_length) == 4 else 65)
                if len(history) >= 2:
                    rear_points = self._rear_trajectory_points(history, rear_half_length)
                    trajectory = self._smooth_trajectory(rear_points)
                    track_color = self._track_annotation_color(track_id)
                    cv2.polylines(summary_frame, [trajectory], False, (0, 0, 0), summary_thickness(8), cv2.LINE_AA)
                    cv2.polylines(summary_frame, [trajectory], False, (255, 255, 255), summary_thickness(6), cv2.LINE_AA)
                    cv2.polylines(summary_frame, [trajectory], False, track_color, summary_thickness(3), cv2.LINE_AA)
                observation = self.car_last_observations.get(track_id, {})
                bbox = observation.get("bbox", [])
                polygon = observation.get("polygon", [])
                best_observation = self.car_best_observations.get(track_id)
                if best_observation and len(bbox) == 4 and len(best_observation.get("bbox", [])) == 4:
                    best_bbox = best_observation["bbox"]
                    best_polygon = np.asarray(best_observation.get("polygon", []), dtype=np.float32)
                    if len(best_polygon) >= 3:
                        best_width = max(1, best_bbox[2] - best_bbox[0])
                        best_height = max(1, best_bbox[3] - best_bbox[1])
                        polygon = np.column_stack((
                            bbox[0] + (best_polygon[:, 0] - best_bbox[0]) / best_width * (bbox[2] - bbox[0]),
                            bbox[1] + (best_polygon[:, 1] - best_bbox[1]) / best_height * (bbox[3] - bbox[1]),
                        )).astype(np.int32).tolist()
                if len(bbox) == 4:
                    cv2.rectangle(summary_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  (255, 255, 0), summary_thickness(4))
                if polygon:
                    cv2.polylines(summary_frame, [np.asarray(polygon, dtype=np.int32)],
                                  True, (0, 255, 255), summary_thickness(3), cv2.LINE_AA)
                center = observation.get("center", [])
                if len(center) == 2:
                    track_color = self._track_annotation_color(track_id)
                    angle_radians = math.radians(float(observation.get("angle", 0)))
                    # 直接用畫面上水藍色外框的長度，不用另外取樣
                    half_length = (max(20, max(bbox[2] - bbox[0], bbox[3] - bbox[1])) / 2
                                    if len(bbox) == 4 else 65)
                    angle_delta = (int(math.cos(angle_radians) * half_length),
                                   int(math.sin(angle_radians) * half_length))
                    angle_start = (center[0] - angle_delta[0], center[1] - angle_delta[1])
                    angle_end = (center[0] + angle_delta[0], center[1] + angle_delta[1])
                    # 紅點標示車尾位置，與角度線非箭頭那端（軌跡點）重疊
                    cv2.circle(summary_frame, angle_start, summary_thickness(8), (0, 0, 255), -1)
                    cv2.arrowedLine(summary_frame, angle_start, angle_end,
                                    (0, 0, 0), summary_thickness(10), cv2.LINE_AA, tipLength=0.2)
                    cv2.arrowedLine(summary_frame, angle_start, angle_end,
                                    (255, 255, 255), summary_thickness(8), cv2.LINE_AA, tipLength=0.2)
                    cv2.arrowedLine(summary_frame, angle_start, angle_end,
                                    track_color, summary_thickness(5), cv2.LINE_AA, tipLength=0.2)
                    heading_angle = int(round(float(observation.get("angle", 0)))) % 360
                    cv2.putText(summary_frame, f"{heading_angle}°",
                                (angle_end[0] + summary_thickness(8), angle_end[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, summary_font(0.75), track_color,
                                summary_thickness(2), cv2.LINE_AA)
                if len(bbox) == 4:
                    label_x, label_y = bbox[0], max(30, bbox[1] - 10)
                else:
                    center = observation.get("center", [20, 30])
                    label_x, label_y = center[0], max(30, center[1])
                speed = observation.get("speed", 0)
                angle = observation.get("drift_angle", observation.get("angle", 0))
                cv2.putText(summary_frame, f"CAR {track_id}  {speed:.1f}  {abs(round(angle))}°",
                            (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, summary_font(0.55),
                            (0, 255, 255), summary_thickness(1), cv2.LINE_AA)
            overlay = np.zeros_like(summary_frame)
            cv2.rectangle(overlay, (0, 0), (summary_frame.shape[1], summary_frame.shape[0]), (0, 0, 0), -1)
            summary_frame = cv2.addWeighted(summary_frame, 0.35, overlay, 0.65, 0)
            # Draw all trajectory annotations after dimming the base image, so they stay on top.
            for track_id, history in self.car_history.items():
                if len(history) < 3:
                    continue
                best_bbox = self.car_best_observations.get(track_id, {}).get("bbox", [])
                rear_half_length = (max(20, max(best_bbox[2] - best_bbox[0], best_bbox[3] - best_bbox[1])) / 2
                                    if len(best_bbox) == 4 else 110)
                rear_points = self._rear_trajectory_points(history, rear_half_length)
                trajectory = self._smooth_trajectory(rear_points)
                track_color = self._track_annotation_color(track_id)
                cv2.polylines(summary_frame, [trajectory], False, (0, 0, 0), summary_thickness(12), cv2.LINE_AA)
                cv2.polylines(summary_frame, [trajectory], False, (255, 255, 255), summary_thickness(8), cv2.LINE_AA)
                cv2.polylines(summary_frame, [trajectory], False, track_color, summary_thickness(5), cv2.LINE_AA)
                label_step = max(1, len(history) // 8)
                # 前後五筆追蹤資料不畫逐幀角度線，避免起步與結束時的方向不穩定。
                for point_index in range(5, max(5, len(history) - 5), label_step):
                    previous = history[point_index - 1]
                    current = history[point_index]
                    frame_delta = max(1, current[2] - previous[2])
                    distance = math.hypot(
                        current[0][0] - previous[0][0],
                        current[0][1] - previous[0][1],
                    )
                    point_speed = distance * self.video_fps / frame_delta
                    movement_angle = math.degrees(math.atan2(
                        current[0][1] - previous[0][1],
                        current[0][0] - previous[0][0],
                    )) % 360
                    drift_angle = abs(current[1] - movement_angle)
                    if drift_angle > 180:
                        drift_angle = 360 - drift_angle
                    label_point = tuple(trajectory[point_index])
                    angle_radians = math.radians(float(current[1]))
                    # 軌跡點即為車尾，角度線非箭頭那端與其重疊，箭頭端延伸車長距離代表車頭
                    angle_delta = (int(math.cos(angle_radians) * rear_half_length * 2),
                                   int(math.sin(angle_radians) * rear_half_length * 2))
                    angle_start = label_point
                    angle_end = (label_point[0] + angle_delta[0], label_point[1] + angle_delta[1])
                    cv2.circle(summary_frame, label_point, summary_thickness(8), (0, 0, 0), -1)
                    cv2.circle(summary_frame, label_point, summary_thickness(5), (0, 0, 255), -1)
                    cv2.arrowedLine(summary_frame, angle_start, angle_end,
                                    (0, 0, 0), summary_thickness(14), cv2.LINE_AA, tipLength=0.2)
                    cv2.arrowedLine(summary_frame, angle_start, angle_end,
                                    (255, 255, 255), summary_thickness(10), cv2.LINE_AA, tipLength=0.2)
                    cv2.arrowedLine(summary_frame, angle_start, angle_end,
                                    track_color, summary_thickness(7), cv2.LINE_AA, tipLength=0.2)
                    angle_text = f"{abs(round(drift_angle))}°"
                    text_origin = (angle_end[0] + summary_thickness(12), angle_end[1])
                    cv2.putText(summary_frame, angle_text, text_origin,
                                cv2.FONT_HERSHEY_SIMPLEX, summary_font(0.8), (0, 0, 0),
                                summary_thickness(4), cv2.LINE_AA)
                    cv2.putText(summary_frame, angle_text, text_origin,
                                cv2.FONT_HERSHEY_SIMPLEX, summary_font(0.8), track_color,
                                summary_thickness(2), cv2.LINE_AA)
                    speed_text = f"{point_speed:.1f}"
                    speed_origin = (label_point[0] + summary_thickness(15),
                                    label_point[1] + summary_thickness(42))
                    cv2.putText(summary_frame, speed_text, speed_origin,
                                cv2.FONT_HERSHEY_SIMPLEX, summary_font(0.75), (0, 0, 0),
                                summary_thickness(4), cv2.LINE_AA)
                    cv2.putText(summary_frame, speed_text, speed_origin,
                                cv2.FONT_HERSHEY_SIMPLEX, summary_font(0.75), (255, 255, 255),
                                summary_thickness(2), cv2.LINE_AA)
            cv2.rectangle(summary_frame, summary_ui_point(20, 20), summary_ui_point(560, 112),
                          (0, 0, 0), -1)
            cv2.putText(summary_frame, "FINAL SCORE", summary_ui_point(36, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, summary_font(1.35), (0, 255, 255),
                        summary_thickness(3), cv2.LINE_AA)
            cv2.putText(summary_frame, "R: replay | Q/ENTER: close", summary_ui_point(36, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, summary_font(0.7), (255, 255, 255),
                        summary_thickness(2), cv2.LINE_AA)

            row_y = 140
            if self.car_scores:
                for track_id in sorted(self.car_scores):
                    events = self.car_events.get(track_id, {"crashes": 0, "spins": 0})
                    detail = self.car_details.get(track_id, {})
                    summary = (f"Car {track_id}: SCORE {self.car_scores[track_id]:.1f}/100 | "
                               f"CRASH {events['crashes']} | SPIN {events['spins']}")
                    cv2.putText(summary_frame, summary, summary_ui_point(40, row_y),
                                cv2.FONT_HERSHEY_SIMPLEX, summary_font(0.95), (255, 255, 255),
                                summary_thickness(3), cv2.LINE_AA)
                    breakdown = (f"Route {detail.get('line_score', 0):.1f}/40  "
                                 f"Angle {detail.get('angle_score', 0):.1f}/20  "
                                 f"Speed {detail.get('speed_score', 0):.1f}/10  "
                                 f"Stability {detail.get('speed_stability_score', 0) + detail.get('angle_stability_score', 0):.1f}/30")
                    cv2.putText(summary_frame, breakdown, summary_ui_point(40, row_y + 24),
                                cv2.FONT_HERSHEY_SIMPLEX, summary_font(0.7), (180, 255, 180),
                                summary_thickness(2), cv2.LINE_AA)
                    row_y += 52
            else:
                cv2.putText(summary_frame, "No vehicle was detected", summary_ui_point(40, row_y),
                            cv2.FONT_HERSHEY_SIMPLEX, summary_font(0.9), (0, 165, 255),
                            summary_thickness(2), cv2.LINE_AA)

            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_results")
            os.makedirs(results_dir, exist_ok=True)
            summary_frame = cv2.resize(summary_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
            summary_path = os.path.join(results_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(summary_path, summary_frame)
            print(f"[分數畫面] 已保存: {summary_path}")
            self._save_score_results()
            cv2.imshow(display_window, summary_frame)
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('r'):
                    cv2.destroyWindow(display_window)
                    self._reset_analysis_state()
                    print("[重播] 已清除上一輪分數，重新分析影片...")
                    return self.process_video()
                if key in (ord('q'), 13):
                    break

        cv2.destroyAllWindows()
        print("影片分析結束，已顯示最終分數。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="遙控甩尾影片評分程式")
    parser.add_argument("--mode", choices=("score", "train-track", "train-track-only", "train-vehicle", "train-vehicle-only", "auto-augment", "curate-validation"), default="score",
                        help="功能模式：score 記分、train-track 跑道訓練、train-track-only 使用現有跑道資料訓練、train-vehicle 手動標記、train-vehicle-only 使用現有資料訓練、auto-augment 自動擴充資料、curate-validation 挑選驗證集")
    parser.add_argument("--sample-frames", type=int, default=5, help="每 X 幀計算一次成績")
    parser.add_argument("--track-interval", type=int, default=1, help="每幾幀執行一次車輛辨識")
    parser.add_argument("--track-merge-iou", type=float, default=0.35,
                        help="同車重複框的 IoU 合併門檻，越低越容易合併")
    parser.add_argument("--track-merge-center-ratio", type=float, default=0.55,
                        help="同車中心距離相對框尺寸比例，越高越容易合併")
    parser.add_argument("--track-merge-min-distance", type=float, default=30.0,
                        help="同車中心距離的像素下限")
    parser.add_argument("--track-merge-containment", type=float, default=0.65,
                        help="同車框包含比例門檻，越低越容易合併")
    parser.add_argument("--track-model", default="track_seg_model.pt", help="跑道模型路徑")
    parser.add_argument("--car-model", default=None, help="車輛模型路徑")
    parser.add_argument("--video-dir", default=None, help="自動擴充模式使用的影片資料夾路徑")
    parser.add_argument("--augment-conf", type=float, default=0.12, help="自動擴充模式使用的最低 confidence 閾值")
    parser.add_argument("--augment-imgsz", type=int, default=1280,
                        help="自動標記推論尺寸；1280 較精細，640 較快")
    parser.add_argument("--augment-max-candidates", type=int, default=20,
                        help="每個影格最多推論的候選框數；達到上限就跳過，例如 5")
    parser.add_argument("--augment-max-per-video", type=int, default=20,
                        help="每支影片最多保存的自動標記樣本數；0 表示不限制")
    parser.add_argument("--augment-duplicate-threshold", type=float, default=0.0,
                        help="連續樣本灰階差異低於此值時跳過；0 表示不做相似畫面去重")
    parser.add_argument("--augment-max-labels", "--augment-skip-labels", dest="augment_max_labels",
                        type=int, default=None,
                        help="自動擴充只保留不超過此數量的標記，例如 4 表示保留 1 到 4 個標記")
    parser.add_argument("--augment-exact-labels", type=int, default=None,
                        help="自動擴充只保留剛好指定數量的標記，例如 4")
    parser.add_argument("--augment-rounds", type=int, default=1,
                        help="自動擴充重複輪數；1=一次，0=持續執行直到按 Ctrl+C")
    parser.add_argument("--augment-merge-iou", type=float, default=0.25,
                        help="自動擴充合併同車遮罩的 IoU 門檻，越低越容易合併")
    parser.add_argument("--augment-merge-center-ratio", type=float, default=0.9,
                        help="自動擴充合併同車遮罩的中心距離比例門檻，越高越容易合併")
    parser.add_argument("--augment-merge-min-distance", type=float, default=60.0,
                        help="自動擴充合併同車遮罩的最小中心距離門檻（像素）")
    parser.add_argument("--augment-merge-containment", type=float, default=0.30,
                        help="自動擴充合併同車遮罩的包含比例門檻，越低越容易合併")
    parser.add_argument("--augment-train", action="store_true",
                        help="自動標記完成後訓練模型；預設只保存標註，不訓練")
    parser.add_argument("--vehicle-class", type=int, default=0,
                        help="車輛類別 ID；自訂遙控車模型通常為 0，COCO car=2，-1=全部")
    learning_group = parser.add_mutually_exclusive_group()
    learning_group.add_argument("--learn", dest="enable_learning", action="store_true",
                                help="手繪完成後自動訓練模型")
    learning_group.add_argument("--no-learn", dest="enable_learning", action="store_false",
                                help="不訓練模型，只保存手繪資料")
    parser.set_defaults(enable_learning=False)
    parser.add_argument("--train-vehicle", action="store_true",
                        help="進入遙控車框選與訓練模式")
    parser.add_argument("--vehicle-epochs", type=int, default=50,
                        help="遙控車模型訓練 epochs")
    parser.add_argument("--track-epochs", type=int, default=20,
                        help="跑道模型訓練 epochs")
    parser.add_argument("--vehicle-sample-frames", type=int, default=10,
                        help="每幾幀抽一張供遙控車標註")
    parser.add_argument("--curate-labels", type=int, default=None,
                        help="人工驗證挑選時，只顯示指定數量的車輛標註，例如 2")
    parser.add_argument("--curate-all", action="store_true",
                        help="人工驗證挑選時顯示全部圖片，包含之前已處理的圖片")
    parser.add_argument("--images-after", default=None,
                        help="只處理此時間之後加入的圖片，例如 20260918_010000")
    parser.add_argument("--review-labels", action="store_true",
                        help="檢查既有標註並補畫漏標車輛")
    parser.add_argument("--review-track-labels", action="store_true",
                        help="檢查跑道訓練圖片的護欄與得分區標註")
    parser.add_argument("--review-split", choices=("train", "val", "all"), default="all",
                        help="既有標註檢查範圍，預設 train 與 val")
    args = parser.parse_args()

    try:
        images_after = _parse_images_after(args.images_after)
    except ValueError as error:
        print(f"[錯誤] {error}")
        sys.exit(1)

    selected_mode = "train-vehicle" if args.train_vehicle else args.mode

    if args.review_labels:
        review_existing_labels(args.review_split, images_after=images_after)
        sys.exit()

    if args.review_track_labels:
        review_track_training_labels()
        sys.exit()

    if selected_mode == "curate-validation":
        if args.curate_labels is not None and args.curate_labels < 0:
            print("[錯誤] --curate-labels 不可小於 0。")
            sys.exit(1)
        if not curate_validation_data(args.curate_labels, unseen_only=not args.curate_all,
                          images_after=images_after):
            sys.exit(1)
        sys.exit()

    if selected_mode == "train-vehicle-only":
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vehicle_seg_training")
        print("[車輛訓練] 使用現有 vehicle_seg_training 資料，不重新標記影片。")
        if not _train_vehicle_model(dataset_dir, args.vehicle_epochs):
            sys.exit(1)
        sys.exit()

    if selected_mode == "train-track-only":
        training_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track_training")
        track_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.track_model)
        if not _train_track_model(training_root, track_model_path, args.track_epochs):
            sys.exit(1)
        sys.exit()

    if selected_mode == "auto-augment":
        if args.augment_rounds < 0:
            print("[錯誤] --augment-rounds 不可小於 0。")
            sys.exit(1)
        if args.augment_max_labels is not None and args.augment_max_labels < 1:
            print("[錯誤] --augment-max-labels 必須大於 0。")
            sys.exit(1)
        if args.augment_exact_labels is not None and args.augment_exact_labels < 1:
            print("[錯誤] --augment-exact-labels 必須大於 0。")
            sys.exit(1)
        if args.augment_max_labels is not None and args.augment_exact_labels is not None:
            print("[錯誤] --augment-max-labels 與 --augment-exact-labels 不可同時使用。")
            sys.exit(1)
        if args.augment_max_candidates < 1:
            print("[錯誤] --augment-max-candidates 必須大於 0。")
            sys.exit(1)
        if args.augment_max_per_video < 0:
            print("[錯誤] --augment-max-per-video 不可小於 0。")
            sys.exit(1)
        if args.augment_duplicate_threshold < 0:
            print("[錯誤] --augment-duplicate-threshold 不可小於 0。")
            sys.exit(1)
        augment_conf = getattr(args, "augment_conf", 0.12)
        round_index = 0
        try:
            while args.augment_rounds == 0 or round_index < args.augment_rounds:
                round_index += 1
                if args.video_dir:
                    if not os.path.isdir(args.video_dir):
                        print(f"[錯誤] 影片資料夾不存在: {args.video_dir}")
                        sys.exit(1)
                    video_paths = [
                        os.path.join(args.video_dir, name)
                        for name in sorted(os.listdir(args.video_dir))
                        if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv"))
                    ]
                else:
                    video_paths = select_video_files()
                if not video_paths:
                    print(f"[自動擴充] 第 {round_index} 輪沒有選取影片，停止。")
                    break
                print(f"[自動擴充] 第 {round_index} 輪匯入 {len(video_paths)} 支影片")
                completed = auto_augment_vehicle_data(
                    video_paths,
                    epochs=args.vehicle_epochs,
                    sample_every=max(1, args.vehicle_sample_frames),
                    min_conf=augment_conf,
                    max_per_video=args.augment_max_per_video or sys.maxsize,
                    duplicate_threshold=args.augment_duplicate_threshold,
                    max_label_count=args.augment_max_labels,
                    exact_label_count=args.augment_exact_labels,
                    train_after_capture=args.augment_train,
                    inference_imgsz=max(320, args.augment_imgsz),
                    max_candidates=args.augment_max_candidates,
                    merge_iou=args.augment_merge_iou,
                    merge_center_ratio=args.augment_merge_center_ratio,
                    merge_min_distance=args.augment_merge_min_distance,
                    merge_containment=args.augment_merge_containment,
                )
                if not completed:
                    print("[自動擴充] 本輪沒有新增樣本，停止循環。")
                    break
        except KeyboardInterrupt:
            print("\n[自動擴充] 已按 Ctrl+C，停止循環。")
        sys.exit()

    # 匯入影片：透過 UI 選取視窗取得檔案路徑
    video_path = select_video_file()

    if not video_path:
        print("[錯誤] 未選取任何影片檔案，程式已結束。")
        sys.exit()

    print(f"[已匯入影片]: {video_path}")

    if selected_mode == "train-vehicle":
        collect_vehicle_training_data(video_path, args.vehicle_epochs, args.vehicle_sample_frames,
                                      train_after_save=args.enable_learning)
        sys.exit()

    trained_car_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rc_car_model.pt")
    use_trained_model = os.path.exists(trained_car_model)
    car_model_path = args.car_model or (trained_car_model if use_trained_model else "yolov8n-seg.pt")

    # 強制使用自訂遙控車模型時，class ID 必須為 0；否則 YOLO 會查找錯誤類別並直接漏檢
    vehicle_class = 0 if (use_trained_model or args.car_model is not None) else args.vehicle_class
    if use_trained_model:
        print(f"[車輛模型] 使用已訓練模型: {trained_car_model}")
        print(f"[車輛模型] 強制 class_id = {vehicle_class}")

    merge_values = (
        args.track_merge_iou,
        args.track_merge_center_ratio,
        args.track_merge_containment,
    )
    if any(value < 0 or value > 1 for value in merge_values) or args.track_merge_min_distance < 0:
        print("[錯誤] 追蹤合併參數範圍無效：比例需介於 0 與 1，距離不可小於 0。")
        sys.exit(1)

    # 建立判分系統實體
    analyzer = DriftJudgeSystem(
        video_path=video_path,
        track_model_path=args.track_model,
        sample_frames=args.sample_frames,
        track_interval=args.track_interval,
        track_merge_iou=args.track_merge_iou,
        track_merge_center_ratio=args.track_merge_center_ratio,
        track_merge_min_distance=args.track_merge_min_distance,
        track_merge_containment=args.track_merge_containment,
        enable_learning=args.enable_learning or selected_mode == "train-track",
        car_model_path=car_model_path,
        vehicle_class=vehicle_class
    )
    
    # 1. 方案 A 自動辨識 + 方案 B 介面互動確認
    try:
        if analyzer.setup_zones_interactive():
            if selected_mode == "score":
                analyzer.process_video()
            else:
                print("[跑道訓練] 標註完成；按 Enter 接受手繪結果後，會保存並訓練跑道模型。")
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n[系統] 使用者中止分析，程式已安全結束。")