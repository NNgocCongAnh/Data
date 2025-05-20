import os
import sys
from typing import Dict, List, Union
from ultralytics import YOLO
import torch
import glob

# Import module cfg từ thư mục cha
from .. import cfg

class YOLOPredictor:
    """Quản lý quá trình dự đoán bằng mô hình YOLOv8."""
    
    def __init__(self, model_path=None):
        """
        Khởi tạo YOLOPredictor sử dụng cấu hình từ cfg.py.
        
        Args:
            model_path (str, optional): Đường dẫn đến file mô hình cụ thể. Nếu None, sẽ sử dụng cfg.TRAIN_CONFIG["model_path"].
        """
        # Xác định thư mục gốc của dự án từ cfg
        self.project_root = cfg.PROJECT_ROOT
        
        # Đường dẫn đến file mô hình
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = os.path.join(self.project_root, cfg.TRAIN_CONFIG["model_path"])
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"File mô hình không tồn tại: {self.model_path}")
        
        # Khởi tạo mô hình
        self.model = YOLO(self.model_path)
        
        # Đường dẫn file YAML (dùng cho validation)
        self.yaml_path = os.path.join(self.project_root, cfg.TRAIN_CONFIG["yaml_path"])
        if not os.path.exists(self.yaml_path):
            print(f"Cảnh báo: File YAML không tồn tại: {self.yaml_path}")
        
        # Cấu hình dự đoán
        self.predict_params = cfg.PREDICT_CONFIG["predict_params"]
        
        print(f"Thư mục gốc của dự án: {self.project_root}")
        print(f"Đường dẫn mô hình: {self.model_path}")
        print(f"Đường dẫn YAML: {self.yaml_path}")

    def validate(self) -> None:
        """Validate mô hình trên tập test."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Sử dụng {device.upper()}")
        
        self.model.val(
            data=self.yaml_path,
            imgsz=self.predict_params["imgsz"],
            batch=self.predict_params["batch"],
            device=device,
            split="test"
        )
        print("✅ Validation hoàn tất.")

    def predict(self, source: Union[str, List[str]] = None) -> None:
        """
        Dự đoán trên ảnh hoặc thư mục ảnh.
        
        Args:
            source (Union[str, List[str]], optional): Đường dẫn đến ảnh, danh sách ảnh, hoặc thư mục.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Sử dụng {device.upper()}")
        
        # Xử lý source
        if source is None:
            source = os.path.join(cfg.get_output_images_dir(), "test")
            print(f"Không cung cấp source, sử dụng mặc định: {source}")
        
        if isinstance(source, str) and os.path.isdir(source):
            # Tìm kiếm các file ảnh với nhiều định dạng và không phân biệt chữ hoa/thường
            image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(source, ext)))
            
            if not image_files:
                raise FileNotFoundError(f"Không tìm thấy file ảnh (jpg/jpeg) trong thư mục: {source}")
        elif isinstance(source, str) and os.path.isfile(source):
            image_files = [source]
        elif isinstance(source, list):
            image_files = source
        else:
            raise ValueError("Source phải là đường dẫn file, thư mục, hoặc danh sách file ảnh.")
        
        # Kiểm tra sự tồn tại của các file ảnh
        for img in image_files:
            if not os.path.exists(img):
                raise FileNotFoundError(f"File ảnh không tồn tại: {img}")
        
        # Dự đoán
        results = self.model.predict(
            source=image_files,
            save=self.predict_params["save"],
            save_txt=self.predict_params["save_txt"],
            save_conf=self.predict_params["save_conf"],
            imgsz=self.predict_params["imgsz"],
            device=device
        )
        
        print(f"✅ Đã dự đoán trên {len(image_files)} ảnh. Kết quả lưu tại: {results[0].save_dir}")

if __name__ == "__main__":
    try:
        predictor = YOLOPredictor()
        
        # Ví dụ dự đoán trên một ảnh
        train_img_dir = os.path.join(cfg.get_output_images_dir(), "train")
        if os.path.exists(train_img_dir):
            image_files = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
            if image_files:
                image_path = os.path.join(train_img_dir, image_files[0])
                print(f"Dự đoán trên ảnh: {image_path}")
                predictor.predict(source=image_path)
        
        # Ví dụ validate trên tập test
        predictor.validate()
        
        # Ví dụ dự đoán trên toàn bộ thư mục test
        predictor.predict()  # Mặc định dùng data/images/test
    except Exception as e:
        print(f"❌ Lỗi: {e}")
