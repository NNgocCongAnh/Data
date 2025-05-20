"""
Module xử lý việc chuẩn bị dữ liệu cho quá trình huấn luyện.

Quy trình:
1. Copy các file cấu hình (classes.txt, notes.json) từ thư mục đầu vào sang thư mục đầu ra
2. Chia dữ liệu thành train/val/test và copy vào thư mục đầu ra
3. Tùy chọn: Thực hiện augmentation trên dữ liệu train
"""

import os
import random
import shutil
import sys
import importlib.util
from typing import List, Tuple

# Import module cfg từ thư mục cha
from .. import cfg

class DataPreparer:
    """Quản lý quá trình chuẩn bị dữ liệu."""
    
    def __init__(self, input_dir=None):
        """
        Khởi tạo DataPreparer.
        
        Args:
            input_dir (str, optional): Thư mục đầu vào. Nếu không được chỉ định, sẽ sử dụng cfg.get_input_dir().
        """
        self.input_dir = input_dir or cfg.get_input_dir()
        print(f"Thư mục đầu vào: {self.input_dir}")
        print(f"Thư mục đầu ra: {cfg.get_output_dir()}")
        
        # Kiểm tra sự tồn tại của thư mục đầu vào
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Thư mục đầu vào không tồn tại: {self.input_dir}")
        
        # Đường dẫn đến thư mục ảnh và nhãn đầu vào
        self.input_images_dir = os.path.join(self.input_dir, cfg.INPUT_DATA_CONFIG["images_subdir"])
        self.input_labels_dir = os.path.join(self.input_dir, cfg.INPUT_DATA_CONFIG["labels_subdir"])
        self.classes_file = os.path.join(self.input_dir, cfg.INPUT_DATA_CONFIG["classes_file"])
        self.notes_file = os.path.join(self.input_dir, cfg.INPUT_DATA_CONFIG["notes_file"])
        
        # Kiểm tra sự tồn tại của các file và thư mục cần thiết
        required_files = [
            (self.classes_file, "File classes"),
            (self.notes_file, "File notes"),
            (self.input_images_dir, "Thư mục ảnh đầu vào"),
            (self.input_labels_dir, "Thư mục nhãn đầu vào")
        ]
        
        for path, desc in required_files:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{desc} không tồn tại: {path}")
    
    def _create_output_directories(self) -> None:
        """Tạo các thư mục đầu ra."""
        # Tạo thư mục đầu ra chính
        os.makedirs(cfg.get_output_dir(), exist_ok=True)
        
        # Tạo thư mục con images và labels
        os.makedirs(cfg.get_output_images_dir(), exist_ok=True)
        os.makedirs(cfg.get_output_labels_dir(), exist_ok=True)
        
        # Tạo thư mục train/val/test
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(cfg.get_output_images_dir(), split), exist_ok=True)
            os.makedirs(os.path.join(cfg.get_output_labels_dir(), split), exist_ok=True)
    
    def _copy_config_files(self) -> None:
        """Copy các file cấu hình từ thư mục đầu vào sang thư mục đầu ra."""
        # Copy classes.txt
        shutil.copy(
            self.classes_file,
            os.path.join(cfg.get_output_dir(), os.path.basename(self.classes_file))
        )
        print(f"Đã copy {self.classes_file} -> {os.path.join(cfg.get_output_dir(), os.path.basename(self.classes_file))}")
        
        # Copy notes.json
        shutil.copy(
            self.notes_file,
            os.path.join(cfg.get_output_dir(), os.path.basename(self.notes_file))
        )
        print(f"Đã copy {self.notes_file} -> {os.path.join(cfg.get_output_dir(), os.path.basename(self.notes_file))}")
    
    def _split_dataset(self) -> Tuple[List[str], List[str], List[str]]:
        """Chia dataset thành train, val, test và trả về danh sách file cho mỗi tập.
        Đảm bảo mỗi tập đều có đủ 5 class."""
        # Lấy danh sách file ảnh
        image_files = [f for f in os.listdir(self.input_images_dir) if f.endswith('.jpg')]
        if not image_files:
            raise FileNotFoundError(f"Không tìm thấy file ảnh .jpg trong thư mục: {self.input_images_dir}")
        
        # Ánh xạ file ảnh với các class chứa trong đó
        image_classes = {}
        for img_file in image_files:
            # Tìm file nhãn tương ứng
            label_file = img_file.replace('.jpg', '.txt')
            label_path = os.path.join(self.input_labels_dir, label_file)
            
            # Nếu không có file nhãn, bỏ qua
            if not os.path.exists(label_path):
                print(f"Cảnh báo: Không tìm thấy nhãn cho ảnh {img_file}")
                continue
            
            # Đọc file nhãn để xác định các class
            classes_in_image = set()
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            classes_in_image.add(class_id)
            except Exception as e:
                print(f"Lỗi khi đọc file nhãn {label_file}: {e}")
                continue
            
            # Lưu thông tin class cho ảnh này
            image_classes[img_file] = classes_in_image
        
        # Tạo danh sách ảnh cho mỗi class
        class_images = {i: [] for i in range(5)}  # Giả sử có 5 class từ 0-4
        for img_file, classes in image_classes.items():
            for class_id in classes:
                if 0 <= class_id < 5:
                    class_images[class_id].append(img_file)
        
        # Kiểm tra xem có class nào không có ảnh không
        for class_id, imgs in class_images.items():
            if not imgs:
                print(f"Cảnh báo: Không có ảnh nào cho class {class_id}")
        
        # Khởi tạo các tập dữ liệu
        train_files = set()
        val_files = set()
        test_files = set()
        
        # Với mỗi class, chia ảnh theo tỉ lệ
        for class_id, imgs in class_images.items():
            # Xáo trộn ảnh của class này
            random.shuffle(imgs)
            
            # Tính số lượng ảnh cho mỗi tập
            total = len(imgs)
            test_count = int(total * cfg.OUTPUT_DATA_CONFIG["test_ratio"])
            val_count = int((total - test_count) * cfg.OUTPUT_DATA_CONFIG["val_ratio"] / (1 - cfg.OUTPUT_DATA_CONFIG["test_ratio"]))
            
            # Chia ảnh vào các tập
            class_test = imgs[-test_count:] if test_count > 0 else []
            class_val = imgs[-(test_count + val_count):-test_count] if val_count > 0 else []
            class_train = imgs[:-(test_count + val_count)] if test_count + val_count < total else []
            
            # Đảm bảo mỗi tập có ít nhất một ảnh của class này (nếu có ảnh)
            if imgs and not class_train and not class_val and not class_test:
                # Nếu không đủ ảnh để chia, thêm ảnh vào tất cả các tập
                class_train = class_val = class_test = [imgs[0]]
            elif imgs and not class_train:
                class_train = [imgs[0]]
            elif imgs and not class_val:
                class_val = [imgs[0]]
            elif imgs and not class_test:
                class_test = [imgs[0]]
            
            # Thêm vào các tập
            train_files.update(class_train)
            val_files.update(class_val)
            test_files.update(class_test)
        
        # Chuyển đổi từ set sang list
        train_files_list = list(train_files)
        val_files_list = list(val_files)
        test_files_list = list(test_files)
        
        # In thông tin về phân bố dữ liệu
        total = len(train_files_list) + len(val_files_list) + len(test_files_list)
        print(f"Tổng số ảnh (sau khi chia): {total}")
        print(f"Số ảnh train: {len(train_files_list)} ({len(train_files_list)/total*100:.1f}%)")
        print(f"Số ảnh validation: {len(val_files_list)} ({len(val_files_list)/total*100:.1f}%)")
        print(f"Số ảnh test: {len(test_files_list)} ({len(test_files_list)/total*100:.1f}%)")
        
        # In thông tin về phân bố class trong mỗi tập
        for name, files in [("Train", train_files_list), ("Validation", val_files_list), ("Test", test_files_list)]:
            classes_in_set = set()
            for img in files:
                classes_in_set.update(image_classes.get(img, set()))
            print(f"Các class trong tập {name}: {sorted(list(classes_in_set))}")
        
        return train_files_list, val_files_list, test_files_list
    
    def _copy_files(self, files: List[str], split: str) -> None:
        """Copy các file ảnh và nhãn vào thư mục tương ứng."""
        for f in files:
            # Copy ảnh
            src_img = os.path.join(self.input_images_dir, f)
            dst_img = os.path.join(cfg.get_output_images_dir(), split, f)
            shutil.copy(src_img, dst_img)
            
            # Copy nhãn nếu tồn tại
            label_file = f.replace('.jpg', '.txt')
            src_label = os.path.join(self.input_labels_dir, label_file)
            dst_label = os.path.join(cfg.get_output_labels_dir(), split, label_file)
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
            else:
                print(f"Cảnh báo: Không tìm thấy nhãn cho ảnh {f}")
    
    def _perform_augmentation(self) -> None:
        """Thực hiện tăng cường dữ liệu."""
        print("Bắt đầu quá trình tăng cường dữ liệu...")
        
        # Import module augmentation
        try:
            from src.gen_data.augmented import DatasetAugmenter
            
            # Cấu hình cho quá trình tăng cường
            aug_config = {
                "images_dir": os.path.join(cfg.get_output_images_dir(), "train"),
                "labels_dir": os.path.join(cfg.get_output_labels_dir(), "train"),
                "output_dir": os.path.join(cfg.get_output_dir(), "augmented"),
                "class_names": cfg.get_class_names(),
                "num_images": cfg.AUGMENTATION_CONFIG["num_images"],
                "num_objects_per_image": cfg.AUGMENTATION_CONFIG["num_objects_per_image"],
                "balance_ratio": cfg.AUGMENTATION_CONFIG["balance_ratio"]
            }
            
            # Thực hiện tăng cường
            augmenter = DatasetAugmenter(aug_config)
            augmenter.augment_dataset()
            
            print("✅ Đã hoàn thành quá trình tăng cường dữ liệu.")
            
        except ImportError as e:
            print(f"Lỗi khi import module tăng cường dữ liệu: {e}")
        except Exception as e:
            print(f"Lỗi khi thực hiện tăng cường dữ liệu: {e}")
    
    def _create_yaml(self) -> None:
        """Tạo file cấu hình YAML cho YOLOv8."""
        yaml_path = os.path.join(os.path.dirname(cfg.get_output_dir()), cfg.TRAIN_CONFIG["yaml_path"])
        
        try:
            import yaml
            
            # Đọc danh sách class names từ file classes.txt
            with open(self.classes_file, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f if line.strip()]
            
            yaml_content = {
                "path": cfg.get_output_dir(),
                "train": os.path.join(cfg.OUTPUT_DATA_CONFIG["images_subdir"], "train"),
                "val": os.path.join(cfg.OUTPUT_DATA_CONFIG["images_subdir"], "val"),
                "test": os.path.join(cfg.OUTPUT_DATA_CONFIG["images_subdir"], "test"),
                "nc": len(class_names),
                "names": class_names
            }
            
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_content, f, allow_unicode=True)
            
            print(f"✅ Đã tạo file YAML tại {yaml_path}")
            
        except ImportError:
            print("Cảnh báo: Không thể import module yaml để tạo file cấu hình. Hãy cài đặt: pip install pyyaml")
        except Exception as e:
            print(f"Lỗi khi tạo file YAML: {e}")
    
    def prepare_dataset(self) -> None:
        """Thực hiện toàn bộ quy trình chuẩn bị dữ liệu."""
        print("Bắt đầu chuẩn bị dữ liệu...")
        
        # Tạo thư mục đầu ra
        self._create_output_directories()
        
        # Copy các file cấu hình
        self._copy_config_files()
        
        # Chia và copy dữ liệu
        train_files, val_files, test_files = self._split_dataset()
        self._copy_files(train_files, "train")
        self._copy_files(val_files, "val")
        self._copy_files(test_files, "test")
        
        # Không tự động thực hiện tăng cường dữ liệu
        # self._perform_augmentation()
        
        # Tạo file YAML
        self._create_yaml()
        
        print("✅ Đã hoàn thành quá trình chuẩn bị dữ liệu.")

if __name__ == "__main__":
    try:
        preparer = DataPreparer()
        preparer.prepare_dataset()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
