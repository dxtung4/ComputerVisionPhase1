from ultralytics import YOLO
import os
import yaml
import multiprocessing


def main():
    # --- 1. KHAI BÁO ĐƯỜNG DẪN ---
    BASE_DIR = r"D:/ComputerVision"
    yaml_path = os.path.join(BASE_DIR, "data_phase1.yaml")
    MODEL_WEIGHTS = os.path.join(BASE_DIR, "Models/yolov11m.pt")
    PATH_TEACHER_OUTPUT = os.path.join(BASE_DIR, "Models/teacher_yolov11m_fix")

    # --- 2. TẢI MÔ HÌNH ---
    print("--> Load mô hình YOLOv11m từ:", MODEL_WEIGHTS)
    teacher_model = YOLO(MODEL_WEIGHTS)

    # --- 3. HUẤN LUYỆN ---
    print("--> Bắt đầu huấn luyện Model với batchsize=4 ...")

    teacher_model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=4,
        patience=20,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=2.0,
        translate=0.1,
        scale=0.4,
        shear=1.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        project=os.path.dirname(PATH_TEACHER_OUTPUT),
        name=os.path.basename(PATH_TEACHER_OUTPUT),
        exist_ok=True,
        device=0,        # 🔥 dùng GPU (nếu có)
        workers=2,        # ⚙️ tránh lỗi spawn tiến trình trên Windows
        # --- Cấu hình Validation & Đánh giá ---
        conf = 0.001,  # Giữ nguyên (Tốt cho Recall/mAP).
        iou = 0.45,
        max_det = 1000,
    )

    print(f"✅ Huấn luyện xong! Model lưu tại: {PATH_TEACHER_OUTPUT}/weights/best.pt")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
