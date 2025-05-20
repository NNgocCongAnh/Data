"""
Tập tin cấu hình cho dự án phân loại bài tập
"""

import os

# Đường dẫn thư mục gốc của dự án
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Cấu hình dữ liệu đầu vào
INPUT_DATA_CONFIG = {
    # Đường dẫn đến thư mục dữ liệu đầu vào (người dùng cần điều chỉnh)
    "input_dir": "data",
    
    # Tên thư mục con chứa ảnh và nhãn trong thư mục đầu vào
    "images_subdir": "images",
    "labels_subdir": "labels",
    
    # Tên file classes và notes trong thư mục đầu vào
    "classes_file": "classes.txt",
    "notes_file": "notes.json"
}

# Cấu hình dữ liệu đầu ra
OUTPUT_DATA_CONFIG = {
    # Đường dẫn đến thư mục dữ liệu đầu ra
    "output_dir": "data",
    
    # Tên thư mục con chứa ảnh và nhãn trong thư mục đầu ra
    "images_subdir": "images",
    "labels_subdir": "labels",
    
    # Tỉ lệ chia dữ liệu
    "test_ratio": 0.2,
    "val_ratio": 0.2
}

# Cấu hình tăng cường dữ liệu (augmentation)
AUGMENTATION_CONFIG = {
    # Có thực hiện augmentation hay không
    "enabled": True,
    
    # Các tham số cho augmentation
    "balance_ratio": 0.9,
    "num_images": 100,
    "num_objects_per_image": 8
}

# Cấu hình huấn luyện
TRAIN_CONFIG = {
    "yaml_path": "project.yaml",
    "model_path": "yolov8s.pt",
    "train_params": {
        "epochs": 75,
        "imgsz": 640,
        "patience": 20,
        "project": "models/project_results",
        "name": "yolov8s",
        "exist_ok": True,
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "lrf": 0.1,
        "cos_lr": True,
        "momentum": 0.937,
        "weight_decay": 0.005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 3.0,
        "cls": 2.0,
        "dfl": 2.0,
        "nbs": 64,
        "dropout": 0.2,
        "hsv_h": 0.02,   # Tăng nhẹ để nhấn mạnh thay đổi màu
        "hsv_s": 0.7,    # Giữ nguyên
        "hsv_v": 0.4,    # Giữ nguyên
        "overlap_mask": False,
        "save": True,
        "plots": True
    }
}

# Cấu hình dự đoán
PREDICT_CONFIG = {
    "predict_params": {
        "imgsz": 640,
        "batch": 16,
        "save": True,
        "save_txt": False,
        "save_conf": False
    }
}
# Các hàm tiện ích để lấy đường dẫn đầy đủ
def get_input_dir():
    """Trả về đường dẫn đầy đủ đến thư mục dữ liệu đầu vào"""
    return os.path.join(PROJECT_ROOT, INPUT_DATA_CONFIG["input_dir"])

def get_output_dir():
    """Trả về đường dẫn đầy đủ đến thư mục dữ liệu đầu ra"""
    return os.path.join(PROJECT_ROOT, OUTPUT_DATA_CONFIG["output_dir"])

def get_input_images_dir():
    """Trả về đường dẫn đầy đủ đến thư mục ảnh đầu vào"""
    return os.path.join(get_input_dir(), INPUT_DATA_CONFIG["images_subdir"])

def get_input_labels_dir():
    """Trả về đường dẫn đầy đủ đến thư mục nhãn đầu vào"""
    return os.path.join(get_input_dir(), INPUT_DATA_CONFIG["labels_subdir"])

def get_output_images_dir():
    """Trả về đường dẫn đầy đủ đến thư mục ảnh đầu ra"""
    return os.path.join(get_output_dir(), OUTPUT_DATA_CONFIG["images_subdir"])

def get_output_labels_dir():
    """Trả về đường dẫn đầy đủ đến thư mục nhãn đầu ra"""
    return os.path.join(get_output_dir(), OUTPUT_DATA_CONFIG["labels_subdir"])

def get_classes_file():
    """Trả về đường dẫn đầy đủ đến file classes.txt"""
    return os.path.join(get_input_dir(), INPUT_DATA_CONFIG["classes_file"])

def get_notes_file():
    """Trả về đường dẫn đầy đủ đến file notes.json"""
    return os.path.join(get_input_dir(), INPUT_DATA_CONFIG["notes_file"])

def get_class_names(custom_input_dir=None):
    """
    Đọc và trả về danh sách tên các lớp từ file classes.txt.
    
    Args:
        custom_input_dir (str, optional): Thư mục đầu vào tùy chỉnh. Nếu được cung cấp,
            sẽ tìm tệp classes.txt trong thư mục này thay vì thư mục mặc định.
    
    Returns:
        List[str]: Danh sách tên các lớp
        
    Raises:
        FileNotFoundError: Nếu không tìm thấy file classes.txt
    """
    if custom_input_dir is not None:
        # Nếu được cung cấp thư mục tùy chỉnh, sử dụng nó
        classes_file = os.path.join(custom_input_dir, INPUT_DATA_CONFIG["classes_file"])
    else:
        # Sử dụng đường dẫn mặc định
        classes_file = get_classes_file()
    
    # Đọc file classes.txt
    with open(classes_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]
        if not class_names:
            raise ValueError(f"File classes.txt tại {classes_file} rỗng")
        return class_names

def set_output_dir(dir_path):
    """Đặt thư mục đầu ra tùy chỉnh"""
    global OUTPUT_DATA_CONFIG
    OUTPUT_DATA_CONFIG["output_dir"] = dir_path
