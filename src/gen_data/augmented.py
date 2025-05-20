import os
import cv2
import numpy as np
import random
import glob
import sys
from collections import Counter
from typing import List, Tuple, Dict, Optional

# Import module cfg từ thư mục cha
from .. import cfg

class DatasetAugmenter:
    """Quản lý quá trình tăng cường dữ liệu cho dataset YOLO."""
    
    def __init__(self, output_dir: str, class_names: List[str] = None, 
                 num_images: int = None, num_objects_per_image: int = None, 
                 balance_ratio: float = None):
        """
        Khởi tạo DatasetAugmenter với các tham số cần thiết.
        
        Args:
            output_dir (str): Thư mục đầu ra (chứa thư mục images và labels)
            class_names (List[str], optional): Tên các lớp. Mặc định lấy từ cfg.
            num_images (int, optional): Số lượng ảnh tạo ra. Mặc định lấy từ cfg.
            num_objects_per_image (int, optional): Số đối tượng mỗi ảnh. Mặc định lấy từ cfg.
            balance_ratio (float, optional): Tỉ lệ cân bằng. Mặc định lấy từ cfg.
        """
        self.output_dir = output_dir
        
        # Kiểm tra xem output_dir có phải là thư mục 'data' hay không
        self.is_data_dir = os.path.basename(os.path.normpath(self.output_dir)) == 'data'
        
        # Nếu class_names được cung cấp, sử dụng nó
        # Nếu không, đọc từ classes.txt trong thư mục đầu ra
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = cfg.get_class_names(custom_input_dir=output_dir)
                
        self.num_images = num_images or cfg.AUGMENTATION_CONFIG["num_images"]
        self.num_objects_per_image = num_objects_per_image or cfg.AUGMENTATION_CONFIG["num_objects_per_image"]
        self.balance_ratio = balance_ratio or cfg.AUGMENTATION_CONFIG["balance_ratio"]
        self.all_objects = []
        
        # Luôn sử dụng images/ và labels/ như thư mục con
        self.images_base_dir = os.path.join(output_dir, 'images')
        self.labels_base_dir = os.path.join(output_dir, 'labels')
        
        # Tạo thư mục nếu chưa tồn tại
        if not os.path.exists(self.images_base_dir):
            os.makedirs(self.images_base_dir, exist_ok=True)
            print(f"Đã tạo thư mục {self.images_base_dir}")
        if not os.path.exists(self.labels_base_dir):
            os.makedirs(self.labels_base_dir, exist_ok=True)
            print(f"Đã tạo thư mục {self.labels_base_dir}")
        
        # Lấy kích thước ảnh từ thư mục đầu vào
        self.img_width, self.img_height = self._get_image_size(self.images_base_dir)

    def _get_image_size(self, images_dir: str) -> Tuple[int, int]:
        """Lấy kích thước ảnh gốc đầu tiên."""
        image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
        if not image_files:
            return 1000, 1414
        img = cv2.imread(image_files[0])
        if img is None:
            print(f"Cảnh báo: Không thể đọc ảnh {image_files[0]}, sử dụng kích thước mặc định")
            return 1000, 1414
        return img.shape[1], img.shape[0]

    def _read_yolo_labels(self, label_path: str) -> List[List[float]]:
        """Đọc file nhãn YOLO."""
        try:
            with open(label_path, 'r') as f:
                return [[float(v) for v in line.strip().split()] for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def _yolo_to_pixel(self, box: List[float]) -> Tuple[int, Tuple[int, int, int, int]]:
        """Chuyển đổi tọa độ YOLO sang pixel."""
        class_id, x_center, y_center, width, height = map(float, box)
        x_center_px = x_center * self.img_width
        y_center_px = y_center * self.img_height
        width_px = width * self.img_width
        height_px = height * self.img_height
        
        x1 = max(0, int(x_center_px - width_px / 2))
        y1 = max(0, int(y_center_px - height_px / 2))
        x2 = min(self.img_width, int(x1 + width_px))
        y2 = min(self.img_height, int(y1 + height_px))
        
        return int(class_id), (x1, y1, x2, y2)

    def _crop_object(self, image: np.ndarray, box: Tuple[int, Tuple[int, int, int, int]]) -> Optional[Tuple[int, np.ndarray, Tuple[int, int]]]:
        """Cắt đối tượng từ ảnh."""
        class_id, (x1, y1, x2, y2) = box
        if x2 <= x1 or y2 <= y1:
            return None
        obj_img = image[y1:y2, x1:x2].copy()
        if obj_img.size == 0:
            return None
        return class_id, obj_img, (x2 - x1, y2 - y1)

    def _pixel_to_yolo(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float, float, float]:
        """Chuyển đổi tọa độ pixel sang YOLO."""
        x_center = (x1 + x2) / 2 / self.img_width
        y_center = (y1 + y2) / 2 / self.img_height
        width = (x2 - x1) / self.img_width
        height = (y2 - y1) / self.img_height
        return x_center, y_center, width, height

    def _create_image_with_objects(self, objects: List[Tuple[int, np.ndarray, Tuple[int, int]]]) -> Tuple[np.ndarray, List[List[float]]]:
        """Tạo ảnh mới bằng cách dán các đối tượng theo chiều dọc."""
        new_image = np.ones((self.img_height, self.img_width, 3), dtype=np.uint8) * 255
        new_labels = []
        current_y = 0
        padding = 10

        for class_id, obj_img, (obj_width, obj_height) in objects:
            h, w = obj_img.shape[:2]
            if w > self.img_width:
                scale = self.img_width / w
                new_w = self.img_width
                new_h = int(h * scale)
                obj_img = cv2.resize(obj_img, (new_w, new_h))
                h, w = new_h, new_w
                
            if current_y + h > self.img_height:
                break
                
            x_offset = max(0, (self.img_width - w) // 2)
            new_image[current_y:current_y + h, x_offset:x_offset + w] = obj_img
            x_center, y_center, width, height = self._pixel_to_yolo(x_offset, current_y, x_offset + w, current_y + h)
            new_labels.append([class_id, x_center, y_center, width, height])
            current_y += h + padding

        return new_image, new_labels

    def _save_image_and_labels(self, image: np.ndarray, labels: List[List[float]], image_name: str, split: str = "custom") -> Tuple[str, str]:
        """
        Lưu ảnh và nhãn mới vào thư mục tương ứng.
        
        Args:
            image: Ảnh cần lưu
            labels: Danh sách nhãn
            image_name: Tên file ảnh
            split: Tên tập dữ liệu (train, val, test) hoặc "custom"
        
        Returns:
            Tuple[str, str]: Đường dẫn đến file ảnh và nhãn đã lưu
        """
        if self.is_data_dir and split != "custom":
            img_dir = os.path.join(self.images_base_dir, split)
            label_dir = os.path.join(self.labels_base_dir, split)
        else:
            img_dir = self.images_base_dir
            label_dir = self.labels_base_dir
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        img_path = os.path.join(img_dir, f'{image_name}.jpg')
        label_path = os.path.join(label_dir, f'{image_name}.txt')

        cv2.imwrite(img_path, image)
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(f"{int(label[0])} {' '.join(f'{x:.6f}' for x in label[1:])}\n")

        return img_path, label_path

    def _get_augmentation_plan(self, class_counts: Counter) -> Dict[int, int]:
        """Tạo kế hoạch tăng cường dữ liệu để cân bằng các lớp."""
        if not class_counts:
            return {}
        max_count = max(class_counts.values())
        target_count = int(max_count * self.balance_ratio)
        return {cid: target_count - count for cid, count in class_counts.items() if count < target_count}

    def _get_specific_image_size(self, img_path: str) -> Tuple[int, int]:
        """Lấy kích thước của một ảnh cụ thể."""
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cảnh báo: Không thể đọc ảnh {img_path}, sử dụng kích thước mặc định")
            return self.img_width, self.img_height
        return img.shape[1], img.shape[0]
        
    def _augment_split(self, split: str = None) -> bool:
        """
        Thực hiện augmentation cho một tập dữ liệu hoặc toàn bộ thư mục images/labels.
        
        Args:
            split: Tên tập dữ liệu (train, val, test) hoặc None nếu không chia tập
        
        Returns:
            bool: True nếu augmentation thành công, False nếu bỏ qua
        """
        if self.is_data_dir and split:
            print(f"\n=== AUGMENTATION CHO TẬP {split.upper()} ===")
            images_dir = os.path.join(self.images_base_dir, split)
            labels_dir = os.path.join(self.labels_base_dir, split)
            split_name = split
        else:
            print("\n=== AUGMENTATION CHO THƯ MỤC TÙY CHỈNH ===")
            images_dir = self.images_base_dir
            labels_dir = self.labels_base_dir
            split_name = "custom"
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Bỏ qua {split_name}: Thư mục {images_dir} hoặc {labels_dir} không tồn tại")
            return False
        
        self.all_objects = []
        image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
        random.shuffle(image_files)
        for img_path in image_files:
            img_name = os.path.basename(img_path).split('.')[0]
            label_path = os.path.join(labels_dir, f'{img_name}.txt')
            image = cv2.imread(img_path)
            if image is None or not os.path.exists(label_path):
                continue
                
            current_width, current_height = image.shape[1], image.shape[0]
            original_width, original_height = self.img_width, self.img_height
            self.img_width, self.img_height = current_width, current_height

            for box in self._read_yolo_labels(label_path):
                pixel_box = self._yolo_to_pixel(box)
                obj = self._crop_object(image, pixel_box)
                if obj:
                    self.all_objects.append(obj)
                    
            self.img_width, self.img_height = original_width, original_height

        print(f"Đã cắt được {len(self.all_objects)} đối tượng từ {split_name}.")
        
        if not self.all_objects:
            print(f"Không có đối tượng để tăng cường trong {split_name}.")
            return False
            
        random.shuffle(self.all_objects)
        class_counts = Counter(obj[0] for obj in self.all_objects)
        aug_plan = self._get_augmentation_plan(class_counts)
        if not aug_plan:
            print(f"{split_name} đã cân bằng.")
            return False

        total_to_add = sum(aug_plan.values())
        print(f"\nKế hoạch tăng cường cho {split_name}:")
        for cid, num in aug_plan.items():
            print(f"- Class {cid} ({self.class_names[cid]}): cần thêm {num} đối tượng")

        new_class_counts = Counter()
        num_images = self.num_images if not self.is_data_dir else max(1, int(self.num_images / 3))
        
        for i in range(num_images):
            sampled_objects = []
            for cid, num in aug_plan.items():
                class_objs = [obj for obj in self.all_objects if obj[0] == cid]
                random.shuffle(class_objs)
                if class_objs:
                    proportion = num / total_to_add
                    num_objs = max(1, int(proportion * self.num_objects_per_image))
                    sampled_objects += random.sample(class_objs, min(num_objs, len(class_objs)))

            sampled_objects = sampled_objects[:self.num_objects_per_image]
            if not sampled_objects:
                continue
                
            random.shuffle(sampled_objects)
            new_class_counts.update(obj[0] for obj in sampled_objects)

            new_img, new_labels = self._create_image_with_objects(sampled_objects)
            self._save_image_and_labels(new_img, new_labels, f"aug_{split_name}_{i+1:04d}", split_name)
            print(f"Đã tạo ảnh aug_{split_name}_{i+1:04d} với {len(new_labels)} đối tượng trong {split_name}")

        print(f"\nThống kê sau tăng cường cho {split_name}:")
        total_objs = sum(class_counts.values()) + sum(new_class_counts.values())
        if total_objs > 0:
            for cid in range(len(self.class_names)):
                count = class_counts.get(cid, 0) + new_class_counts.get(cid, 0)
                print(f"Class {cid} ({self.class_names[cid]}): {count} đối tượng ({count / total_objs * 100:.2f}%)")

        return True

    def augment_dataset(self) -> bool:
        """Tạo dataset mới với các đối tượng được cân bằng."""
        print("=== BẮT ĐẦU QUÁ TRÌNH TĂNG CƯỜNG DỮ LIỆU ===")
        
        success = False
        
        if self.is_data_dir:
            for split in ["train", "val", "test"]:
                if self._augment_split(split):
                    success = True
        else:
            if self._augment_split():
                success = True
        
        if success:
            print("\n=== HOÀN THÀNH QUÁ TRÌNH TĂNG CƯỜNG DỮ LIỆU ===")
            return True
        else:
            print("\n=== KHÔNG CÓ DỮ LIỆU NÀO ĐƯỢC TĂNG CƯỜNG ===")
            return False

if __name__ == "__main__":
    # Lấy thông tin trực tiếp từ cfg.py
    output_dir = cfg.get_output_dir()
    
    print(f"Thư mục hiện tại: {os.getcwd()}")
    print(f"Thư mục đầu ra: {output_dir}")
    
    # Kiểm tra tồn tại thư mục đầu ra
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Thư mục dữ liệu không tồn tại: {output_dir}")
    
    # Khởi tạo DatasetAugmenter trực tiếp
    augmenter = DatasetAugmenter(
        output_dir=output_dir,
        class_names=cfg.get_class_names(),
        num_images=cfg.AUGMENTATION_CONFIG["num_images"],
        num_objects_per_image=cfg.AUGMENTATION_CONFIG["num_objects_per_image"],
        balance_ratio=cfg.AUGMENTATION_CONFIG["balance_ratio"]
    )
    
    # Thực hiện augmentation
    augmenter.augment_dataset()
