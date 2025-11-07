import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from ultralytics import YOLO
import yaml
import shutil

# --- 1. KHAI BÁO CÁC ĐƯỜNG DẪN LOCAL ---
print("--> Khai báo các đường dẫn...")

BASE_DIR = r"D:/ComputerVision"

# Dữ liệu gốc
PATH_F8K_IMAGES = os.path.join(BASE_DIR, "Project/Fisheye8K_all_including_train&test/train/images")
PATH_F8K_LABELS = os.path.join(BASE_DIR, "Project/Fisheye8K_all_including_train&test/train/labels")
PATH_VISDRONE_IMAGES = os.path.join(BASE_DIR, "Project/VisDrone_Dataset/VisDrone2019-DET-train/images")
PATH_VISDRONE_YOLO_LABELS_ORIGINAL = os.path.join(BASE_DIR, "Project/VisDrone_Dataset/VisDrone2019-DET-train/labels")

# Dữ liệu trung gian (nơi lưu kết quả của các bước xử lý).
PATH_VISDRONE_YOLO_LABELS_MAPPED = os.path.join(BASE_DIR, "Project/visdrone_yolo_labels_mapped")
PATH_SYNTHETIC_IMAGES = os.path.join(BASE_DIR, "Project/synthetic_fisheye/images")
PATH_SYNTHETIC_LABELS = os.path.join(BASE_DIR, "Project/synthetic_fisheye/labels")
PATH_DATA_PHASE1 = os.path.join(BASE_DIR, "Project/data_phase1")
PATH_DATA_PHASE1_IMAGES = os.path.join(PATH_DATA_PHASE1, "images")
PATH_DATA_PHASE1_LABELS = os.path.join(PATH_DATA_PHASE1, "labels")

# Validation
PATH_VALIDATION_IMAGES = os.path.join(BASE_DIR, "Project/Fisheye8K_all_including_train&test/test/images")

# Model
PATH_TEACHER_OUTPUT = os.path.join(BASE_DIR, "Models/teacher_yolov11m")
PATH_MODEL_A = os.path.join(PATH_TEACHER_OUTPUT, "weights/best.pt")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "Models/yolov11m.pt")

# Tạo thư mục cần thiết
for path in [
    PATH_VISDRONE_YOLO_LABELS_MAPPED, PATH_SYNTHETIC_IMAGES, PATH_SYNTHETIC_LABELS,
    PATH_DATA_PHASE1_IMAGES, PATH_DATA_PHASE1_LABELS, PATH_TEACHER_OUTPUT
]:
    os.makedirs(path, exist_ok=True)

print("--> Hoàn tất khai báo!")

# --- 2. ÁNH XẠ LẠI NHÃN VISDRONE ---
def remap_visdrone_yolo_labels(input_dir, output_dir):
    print(f"Bắt đầu ánh xạ lại nhãn từ {input_dir}...")
    label_map = {9: 0, 3: 1, 7: 1, 8: 1, 10: 1, 4: 2, 5: 2, 1: 3, 2: 3, 6: 4}
    for label_file in tqdm(os.listdir(input_dir), desc="Remapping VisDrone classes"):
        src_path = os.path.join(input_dir, label_file)
        dst_path = os.path.join(output_dir, label_file)
        new_lines = []
        with open(src_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id in label_map:
                    new_cls_id = label_map[cls_id]
                    coords = " ".join(parts[1:])
                    new_lines.append(f"{new_cls_id} {coords}\n")
        if new_lines:
            with open(dst_path, "w") as f:
                f.writelines(new_lines)

# --- 3. TẠO DỮ LIỆU MẮT CÁ NHÂN TẠO ---
def generate_synthetic_data(img_dir, label_dir, out_img_dir, out_label_dir):
    print("Bắt đầu tạo dữ liệu mắt cá nhân tạo...")
    transform = A.Compose(
        [A.GridDistortion(num_steps=5, distort_limit=0.7, p=1.0)], #Tạo hiệu ứng cong, lồi
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])#tính toán lại tọa độ bbox sau khi làm phồng
    )

    for img_file in tqdm(os.listdir(img_dir), desc="Generating synthetic data"):
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes, class_labels = [], []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_labels.append(int(parts[0]))
                        coords = [float(p) for p in parts[1:]]
                        clipped = np.clip(coords, 0.0, 1.0).tolist()
                        bboxes.append(clipped)

        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            out_img = os.path.join(out_img_dir, img_file)
            cv2.imwrite(out_img, cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR))

            out_lbl = os.path.join(out_label_dir, os.path.splitext(img_file)[0] + ".txt")
            with open(out_lbl, "w") as f:
                for bbox, label in zip(transformed["bboxes"], transformed["class_labels"]):
                    f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        except Exception as e:
            print(f"⚠️ Skipping {img_file}: {e}")

# --- 4. XỬ LÝ DỮ LIỆU ---
remap_visdrone_yolo_labels(PATH_VISDRONE_YOLO_LABELS_ORIGINAL, PATH_VISDRONE_YOLO_LABELS_MAPPED)
generate_synthetic_data(PATH_VISDRONE_IMAGES, PATH_VISDRONE_YOLO_LABELS_MAPPED, PATH_SYNTHETIC_IMAGES, PATH_SYNTHETIC_LABELS)

print("--> Gộp dữ liệu cho Giai đoạn 1...")
for src_folder, dst_folder in [
    (PATH_F8K_IMAGES, PATH_DATA_PHASE1_IMAGES),
    (PATH_F8K_LABELS, PATH_DATA_PHASE1_LABELS),
    (PATH_SYNTHETIC_IMAGES, PATH_DATA_PHASE1_IMAGES),
    (PATH_SYNTHETIC_LABELS, PATH_DATA_PHASE1_LABELS)
]:
    for file in os.listdir(src_folder):
        shutil.copy(os.path.join(src_folder, file), os.path.join(dst_folder, file))
print("--> Hoàn tất Giai đoạn 1!")

# --- 5. TẠO FILE YAML ---
yaml_phase1 = {
    "path": PATH_DATA_PHASE1,
    "train": "images",
    "val": PATH_VALIDATION_IMAGES,
    "nc": 5,
    "names": ["Bus", "Bike", "Car", "Pedestrian", "Truck"]
}
yaml_path = os.path.join(BASE_DIR, "data_phase1.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(yaml_phase1, f)
print("--> File YAML đã tạo:", yaml_path)


