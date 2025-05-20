import os
import numpy as np
from typing import Dict
from ultralytics import YOLO
import torch

class YOLOEvaluator:
    """Quản lý quá trình đánh giá mô hình YOLOv8."""
    
    def __init__(self, config: Dict):
        """
        Khởi tạo YOLOEvaluator với cấu hình.
        
        Args:
            config (Dict): Cấu hình chứa đường dẫn mô hình, YAML, và tham số đánh giá.
        """
        self.config = config
        
        # Xác định thư mục gốc của dự án (lên 2 cấp từ src/eval đến project)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        
        # Kiểm tra file mô hình
        self.model_path = os.path.join(self.project_root, config["model_path"])
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"File mô hình không tồn tại: {self.model_path}")
        
        # Kiểm tra file YAML
        self.yaml_path = os.path.join(self.project_root, config["yaml_path"])
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"File YAML không tồn tại: {self.yaml_path}")
        
        # Khởi tạo mô hình
        self.model = YOLO(self.model_path)
        
        print(f"Thư mục gốc của dự án: {self.project_root}")
        print(f"Đường dẫn mô hình: {self.model_path}")
        print(f"Đường dẫn YAML: {self.yaml_path}")

    def evaluate(self, split: str = "test") -> Dict:
        """
        Đánh giá mô hình trên một tập dữ liệu.
        
        Args:
            split (str): Tập dữ liệu để đánh giá ("train", "val", hoặc "test").
        
        Returns:
            Dict: Chỉ số đánh giá (F1, mAP@0.5, mAP@0.5:0.95, Precision, Recall).
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Sử dụng {device.upper()} để đánh giá trên tập {split}")
        
        metrics = self.model.val(
            data=self.yaml_path,
            imgsz=self.config["eval_params"]["imgsz"],
            batch=self.config["eval_params"]["batch"],
            device=device,
            split=split
        )
        
        return {
            "f1": metrics.box.f1,  # Mảng F1 score cho từng lớp
            "map50": metrics.box.map50,  # mAP@0.5
            "map": metrics.box.map,  # mAP@0.5:0.95
            "precision": metrics.box.p,  # Precision
            "recall": metrics.box.r  # Recall
        }

    def print_metrics(self, split: str, metrics: Dict) -> None:
        """
        In các chỉ số đánh giá.
        
        Args:
            split (str): Tên tập dữ liệu ("train", "val", hoặc "test").
            metrics (Dict): Chỉ số đánh giá.
        """
        print(f"\n{split.capitalize()} Metrics:")
        
        # Tính giá trị trung bình cho các chỉ số
        f1_mean = float(np.mean(metrics['f1'])) if metrics['f1'] is not None and len(metrics['f1']) > 0 else 0.0
        map50_mean = float(np.mean(metrics['map50'])) if metrics['map50'] is not None else 0.0
        map_mean = float(np.mean(metrics['map'])) if metrics['map'] is not None else 0.0
        precision_mean = float(np.mean(metrics['precision'])) if metrics['precision'] is not None and len(metrics['precision']) > 0 else 0.0
        recall_mean = float(np.mean(metrics['recall'])) if metrics['recall'] is not None and len(metrics['recall']) > 0 else 0.0
        
        # In chỉ số tổng quát
        print("Tổng quát:")
        print(f"F1 Score: {f1_mean:.2f}")
        print(f"Precision: {precision_mean:.2f}")
        print(f"Recall: {recall_mean:.2f}")
        print(f"Accuracy (mAP@0.5): {map50_mean:.2%}")
        print(f"Accuracy (mAP@0.5:0.95): {map_mean:.2%}")
        
        # In chi tiết cho từng lớp
        print("\nChi tiết theo lớp:")
        for i, class_name in enumerate(self.config["class_names"]):
            f1 = float(metrics['f1'][i]) if i < len(metrics['f1']) else 0.0
            precision = float(metrics['precision'][i]) if i < len(metrics['precision']) else 0.0
            recall = float(metrics['recall'][i]) if i < len(metrics['recall']) else 0.0
            map50 = float(metrics['map50']) if metrics['map50'] is not None else 0.0  # mAP50 là scalar
            map = float(metrics['map']) if metrics['map'] is not None else 0.0  # mAP là scalar
            print(f"{class_name}:")
            print(f"  F1 Score: {f1:.2f}")
            print(f"  Precision: {precision:.2f}")
            print(f"  Recall: {recall:.2f}")
            print(f"  mAP@0.5: {map50:.2%}")
            print(f"  mAP@0.5:0.95: {map:.2%}")

if __name__ == "__main__":
    config = {
        "model_path": "project7_results/yolov8s/weights/best.pt",
        "yaml_path": "project7.yaml",
        "class_names": ['None', 'Nối', 'Trắc nghiệm', 'Tự luận', 'Điền từ', 'Đặt tính rồi tính'],
        "eval_params": {
            "imgsz": 640,
            "batch": 16
        }
    }
    
    evaluator = YOLOEvaluator(config)
    
    # Đánh giá trên tập train
    train_metrics = evaluator.evaluate(split="train")
    evaluator.print_metrics("train", train_metrics)
    
    # Đánh giá trên tập test
    test_metrics = evaluator.evaluate(split="test")
    evaluator.print_metrics("test", test_metrics)
