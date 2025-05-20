# Hệ thống Xử lý và Huấn luyện YOLO

Hệ thống này cung cấp các công cụ để chuẩn bị dữ liệu, tăng cường dữ liệu, huấn luyện và đánh giá mô hình YOLO từ định dạng PDF và hình ảnh.

## Cài đặt

<<<<<<< HEAD
1. Clone repository:
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

2. Cài đặt các thư viện cần thiết:
=======
1. Cài đặt các thư viện cần thiết:
>>>>>>> 6859a69 ([feat][push_code_1305_2][init YOLO training pipeline])
```bash
pip install -r requirements.txt
```

## Các tác vụ được hỗ trợ

- `pdf_to_image`: Chuyển đổi PDF sang hình ảnh
- `prepare`: Chuẩn bị dữ liệu (chia train/val/test)
- `augment`: Tăng cường dữ liệu
- `train`: Huấn luyện mô hình
- `predict`: Dự đoán với mô hình đã huấn luyện
- `eval`: Đánh giá mô hình
- `all`: Thực hiện tất cả các bước trên

## I. Hướng dẫn thêm dữ liệu

### Trường hợp 1: Bắt đầu từ file PDF

1. **Chuyển đổi PDF sang hình ảnh**:
   ```bash
   python app.py --task pdf_to_image --pdf-path "đường/dẫn/tới/file.pdf" --output-dir image
   ```
   
   Các hình ảnh sẽ được lưu vào thư mục `image/` với tên `page_1.jpeg`, `page_2.jpeg`, v.v.

2. **Gắn nhãn cho dữ liệu sử dụng Label Studio**:
   - Tải và cài đặt Label Studio: [https://labelstud.io/](https://labelstud.io/)
   - Tạo dự án mới trong Label Studio và nhập các hình ảnh từ bước 1
   - Thiết lập giao diện gắn nhãn cho Object Detection (YOLO)
   - Tiến hành gắn nhãn cho các đối tượng trong hình ảnh
   - Xuất dữ liệu dưới dạng "YOLO with Image" từ Label Studio
   - Giải nén file xuất ra và sử dụng đường dẫn đến thư mục giải nén làm đầu vào cho bước tiếp theo

3. **Chuẩn bị dữ liệu** (chia thành train/val/test):
   ```bash
   python app.py --task prepare --input-dir đường/dẫn/tới/thư_mục_từ_label_studio --output-dir data
   ```

4. **Tăng cường dữ liệu**:
   ```bash
   python app.py --task augment --output-dir data
   ```

5. **Huấn luyện mô hình**:
   ```bash
   python app.py --task train
   ```
   
   **Lưu ý quan trọng về thư mục dữ liệu**:
   - Tất cả các tác vụ (prepare, augment, train) đều **cố định thư mục đầu ra là `data/`**, bất kể tham số --output-dir được chỉ định là gì
   - Các tác vụ prepare và augment vẫn cho phép bạn chỉ định thư mục đầu vào qua --input-dir hoặc cấu hình trong cfg.py
   - Tác vụ train luôn sử dụng dữ liệu từ thư mục `data/` hiện có
   - Đây là thiết kế có chủ đích để đảm bảo tính nhất quán của dữ liệu trong toàn bộ quy trình

### Trường hợp 2: Bắt đầu từ dữ liệu YOLO có sẵn

1. **Chuẩn bị cấu trúc thư mục**:
   Cấu trúc thư mục cần có dạng:
   ```
   project-data/
   ├── classes.txt   # Chứa tên các lớp
   ├── notes.json    # (Tùy chọn) Ghi chú về dữ liệu
   ├── images/       # Thư mục chứa hình ảnh (.jpg)
   └── labels/       # Thư mục chứa nhãn (.txt)
   ```

2. **Chuẩn bị dữ liệu** (chia thành train/val/test):
   ```bash
   python app.py --task prepare --input-dir project-data --output-dir data
   ```

3. **Tăng cường dữ liệu**:
   ```bash
   python app.py --task augment --output-dir data
   ```

4. **Huấn luyện mô hình**:
   ```bash
   python app.py --task train --output-dir data
   ```

## II. Hướng dẫn cấu hình thông qua tham số dòng lệnh

Tất cả các cấu hình có thể được điều chỉnh thông qua tham số dòng lệnh. Dưới đây là các tham số phổ biến cho từng tác vụ:

### 1. Tham số chung

- `--input-dir`: Thư mục dữ liệu đầu vào
- `--output-dir`: Thư mục đầu ra

### 2. Chuyển đổi PDF sang hình ảnh

```bash
python app.py --task pdf_to_image --pdf-path "đường/dẫn/tới/file.pdf" --output-dir image --format JPEG
```

Tham số:
- `--pdf-path`: Đường dẫn đến file PDF
- `--format`: Định dạng ảnh đầu ra (JPEG, PNG, BMP)

### 3. Chuẩn bị dữ liệu

```bash
python app.py --task prepare --input-dir project-data --output-dir data --test-ratio 0.2 --val-ratio 0.2
```

Tham số:
- `--test-ratio`: Tỉ lệ dữ liệu test (mặc định: 0.3)
- `--val-ratio`: Tỉ lệ dữ liệu validation (mặc định: 0.2)

### 4. Tăng cường dữ liệu

```bash
python app.py --task augment --output-dir data --num-images 300 --objects-per-image 5 --balance-ratio 0.8
```

Tham số:
- `--num-images`: Tổng số ảnh tăng cường cần tạo
- `--objects-per-image`: Số đối tượng trên mỗi ảnh tăng cường
- `--balance-ratio`: Tỉ lệ cân bằng giữa các lớp

### 5. Huấn luyện mô hình

```bash
python app.py --task train --output-dir data --model yolov8s.pt --epochs 100 --batch-size 16 --img-size 640
```

Tham số:
- `--model`: Đường dẫn đến mô hình pretrained
- `--epochs`: Số epochs huấn luyện
- `--batch-size`: Kích thước batch
- `--img-size`: Kích thước ảnh đầu vào
- `--model-output`: Thư mục lưu mô hình đầu ra

### 6. Dự đoán

```bash
python app.py --task predict --source data/images/test --save-txt --save-conf
```

Tham số:
- `--source`: Nguồn dữ liệu dự đoán (ảnh, thư mục ảnh, video)
- `--save-txt`: Lưu kết quả dạng text
- `--save-conf`: Lưu độ tin cậy trong kết quả
- `--do-validate`: Thực hiện validate trước khi dự đoán

### 7. Đánh giá mô hình

```bash
python app.py --task eval --split test,val --save-metrics --metrics-output metrics/ --best-model
```

Tham số:
- `--split`: Tập dữ liệu cần đánh giá (train, val, test)
- `--save-metrics`: Lưu kết quả đánh giá
- `--metrics-output`: Thư mục lưu kết quả đánh giá
- `--best-model`: Sử dụng file best.pt

### 8. Thực hiện tất cả các bước

```bash
python app.py --task all --input-dir project-data --output-dir data --model yolov8s.pt
```

## III. Hướng dẫn cấu hình thông qua file cfg.py

Nếu bạn muốn thay đổi cấu hình mặc định cho nhiều lần chạy, bạn có thể sửa trực tiếp file `cfg.py`. Dưới đây là các cấu hình phổ biến:

### 1. Cấu hình đường dẫn dữ liệu

```python
# Đường dẫn dữ liệu đầu vào
INPUT_DATA_CONFIG = {
    "input_dir": "project-data",  # Thư mục dữ liệu đầu vào
    "images_subdir": "images",    # Thư mục con chứa hình ảnh
    "labels_subdir": "labels",    # Thư mục con chứa nhãn
    "classes_file": "classes.txt", # File chứa tên các lớp
    "notes_file": "notes.json"    # File ghi chú (tùy chọn)
}

# Đường dẫn dữ liệu đầu ra
OUTPUT_DATA_CONFIG = {
    "output_dir": "data",         # Thư mục dữ liệu đầu ra
    "images_subdir": "images",    # Thư mục con chứa hình ảnh
    "labels_subdir": "labels",    # Thư mục con chứa nhãn
    "test_ratio": 0.3,            # Tỉ lệ dữ liệu test
    "val_ratio": 0.2              # Tỉ lệ dữ liệu validation
}
```

### 2. Cấu hình tăng cường dữ liệu

```python
# Cấu hình tăng cường dữ liệu
AUGMENTATION_CONFIG = {
    "num_images": 300,            # Số lượng ảnh tăng cường
    "num_objects_per_image": 5,   # Số đối tượng trên mỗi ảnh
    "balance_ratio": 0.8          # Tỉ lệ cân bằng giữa các lớp
}
```

### 3. Cấu hình huấn luyện

```python
# Cấu hình huấn luyện
TRAIN_CONFIG = {
    "model_path": "yolov8s.pt",   # Đường dẫn mô hình pretrained
    "yaml_path": "project.yaml",  # Đường dẫn file YAML
    "train_params": {             # Tham số huấn luyện
        "epochs": 100,            # Số epochs
        "batch": 16,              # Kích thước batch
        "imgsz": 640,             # Kích thước ảnh
        "device": "",             # Thiết bị (tự động chọn)
        "workers": 8,             # Số worker
        "project": "runs/train",  # Thư mục lưu kết quả
        "name": "exp",            # Tên thí nghiệm
        "exist_ok": False,        # Ghi đè thư mục đã tồn tại
        "pretrained": True,       # Sử dụng mô hình pretrained
        "optimizer": "auto",      # Loại optimizer
        "verbose": False,         # In thông tin chi tiết
        "seed": 0,                # Seed ngẫu nhiên
        "deterministic": True,    # Đảm bảo kết quả giống nhau
        "single_cls": False,      # Phát hiện đối tượng đơn lớp
        "image_weights": False,   # Sử dụng cân bằng trọng số ảnh
        "rect": False,            # Hình chữ nhật
        "cos_lr": False,          # Cosine LR scheduler
        "close_mosaic": 10,       # Số epochs trước khi đóng mosaic
        "resume": False,          # Tiếp tục huấn luyện
        "amp": True,              # Sử dụng mixed precision
        "fraction": 1.0,          # Tỉ lệ dataset sử dụng
        "profile": False,         # Đo hiệu suất
        "overlap_mask": True,     # Cho phép overlap mask
        "mask_ratio": 4,          # Tỉ lệ mask
        "dropout": 0.0,           # Tỉ lệ dropout
        "val": True,              # Validate trong khi huấn luyện
        "save": True,             # Lưu kết quả
        "save_period": -1,        # Lưu kết quả theo chu kỳ
        "cache": False,           # Cache images
        "patience": 100,          # EarlyStopping patience
        "freeze": None,           # Đóng băng các lớp
        "plots": True,            # Vẽ đồ thị kết quả
    }
}
```

### 4. Cấu hình dự đoán

```python
# Cấu hình dự đoán
PREDICT_CONFIG = {
    "model_path": "best.pt",      # Đường dẫn mô hình
    "predict_params": {           # Tham số dự đoán
        "source": "",             # Nguồn dữ liệu
        "conf": 0.25,             # Ngưỡng tin cậy
        "iou": 0.45,              # Ngưỡng IoU
        "show": False,            # Hiển thị kết quả
        "save": True,             # Lưu kết quả
        "save_txt": False,        # Lưu kết quả dạng text
        "save_conf": False,       # Lưu độ tin cậy
        "show_labels": True,      # Hiển thị nhãn
        "show_conf": True,        # Hiển thị độ tin cậy
        "max_det": 300,           # Số phát hiện tối đa
        "vid_stride": 1,          # Video stride
        "line_width": None,       # Độ rộng đường viền
        "visualize": False,       # Hiển thị feature
        "augment": False,         # Augmented inference
        "agnostic_nms": False,    # Class-agnostic NMS
        "classes": None,          # Lọc theo lớp
        "retina_masks": False,    # Dùng retina masks
        "boxes": True,            # Hiển thị hộp giới hạn
        "imgsz": 640,             # Kích thước ảnh đầu vào
        "batch": 1,               # Số ảnh mỗi batch
    }
}
```

### Cách sử dụng sau khi sửa cfg.py

Sau khi bạn đã thay đổi file `cfg.py`, bạn có thể chạy app với các lệnh đơn giản:

```bash
# Chuẩn bị dữ liệu
python app.py --task prepare

# Tăng cường dữ liệu
python app.py --task augment

# Huấn luyện mô hình
python app.py --task train

# Dự đoán
python app.py --task predict

# Đánh giá mô hình
python app.py --task eval
```

**Lưu ý**: Các tham số dòng lệnh sẽ ghi đè lên cấu hình trong `cfg.py`. Ví dụ:

```bash
# Ghi đè số epochs thành 50 (thay vì giá trị trong cfg.py)
python app.py --task train --epochs 50
```

## Câu hỏi thường gặp

### Q: Cấu trúc thư mục sau khi chuẩn bị dữ liệu như thế nào?

Sau khi chạy `--task prepare`, cấu trúc thư mục sẽ có dạng:
```
data/
├── classes.txt
├── notes.json (nếu có)
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### Q: Tôi có thể cùng lúc thực hiện nhiều thay đổi cấu hình không?

Có, bạn có thể kết hợp nhiều tham số:

```bash
python app.py --task all --input-dir project-data --output-dir data --model yolov8s.pt --epochs 50 --batch-size 8 --num-images 200
```

### Q: Làm thế nào để biết kết quả huấn luyện?

Kết quả huấn luyện được lưu tại thư mục `project_results/yolov8s` (theo cấu hình `project` và `name` trong `TRAIN_CONFIG`). Bạn có thể xem các đồ thị, mô hình đã lưu (trong thư mục `weights`), và kết quả đánh giá tại đó.
