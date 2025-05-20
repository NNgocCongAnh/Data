import os
import random
import shutil
import yaml
import sys
from typing import List, Dict, Tuple
from ultralytics import YOLO
import torch

# Import module cfg từ thư mục cha
from .. import cfg

class YOLOTrainer:
    """Quản lý quá trình huấn luyện mô hình YOLOv8."""
    
    def __init__(self):
        """Khởi tạo YOLOTrainer sử dụng cấu hình từ cfg.py."""
        # Sử dụng thư mục dữ liệu từ cfg
        self.dataset_dir = cfg.get_output_dir()
        
        # Kiểm tra sự tồn tại của thư mục dataset
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Thư mục dataset không tồn tại: {self.dataset_dir}. Hãy chạy prepare_data.py trước.")
        
        self.image_dir = cfg.get_output_images_dir()
        self.label_dir = cfg.get_output_labels_dir()
        
        # Kiểm tra sự tồn tại của thư mục images và labels
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Thư mục ảnh không tồn tại: {self.image_dir}")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Thư mục nhãn không tồn tại: {self.label_dir}")
        
        # Đường dẫn đến file YAML
        self.yaml_path = os.path.join(cfg.PROJECT_ROOT, cfg.TRAIN_CONFIG["yaml_path"])
        
        print(f"Thư mục gốc của dự án: {cfg.PROJECT_ROOT}")
        print(f"Đường dẫn dataset: {self.dataset_dir}")
        print(f"Đường dẫn YAML: {self.yaml_path}")

    def _check_directories(self) -> None:
        """Kiểm tra các thư mục train, val, test."""
        for split in ["train", "val", "test"]:
            img_dir = os.path.join(self.image_dir, split)
            label_dir = os.path.join(self.label_dir, split)
            
            if not os.path.exists(img_dir) or not os.path.exists(label_dir):
                raise FileNotFoundError(f"Thư mục {split} không tồn tại. Hãy chạy prepare_data.py trước.")
            
            image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
            if not image_files:
                raise FileNotFoundError(f"Không tìm thấy file ảnh .jpg trong thư mục: {img_dir}")

    def train_model(self) -> None:
        """Huấn luyện mô hình YOLOv8."""
        # Kiểm tra các thư mục
        self._check_directories()
        
        # Xác định thiết bị
        device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"Sử dụng {'GPU: ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
        
        # Đường dẫn đến file mô hình
        model_path = os.path.join(cfg.PROJECT_ROOT, cfg.TRAIN_CONFIG["model_path"])
        
        # Tải mô hình
        try:
            model = YOLO(model_path)
        except RuntimeError as e:
            if "PytorchStreamReader failed reading zip archive" in str(e):
                print(f"❌ Model file {model_path} bị lỗi. Tải lại...")
                torch.hub.download_url_to_file(
                    'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
                    model_path)
                model = YOLO(model_path)
            else:
                raise e

        # Huấn luyện mô hình
        model.train(
            data=self.yaml_path,
            device=device,
            **cfg.TRAIN_CONFIG["train_params"]
        )
        print("✅ Huấn luyện hoàn tất.")

if __name__ == "__main__":
    try:
        trainer = YOLOTrainer()
        trainer.train_model()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
