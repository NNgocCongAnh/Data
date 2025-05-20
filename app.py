#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml

# Import các module sẵn có
from src.gen_data.prepare_data import DataPreparer
from src.gen_data.augmented import DatasetAugmenter
from src.train.train_yolo import YOLOTrainer
from src.predict.predict_yolo import YOLOPredictor
import cfg

def create_data_yaml(input_dir, output_yaml, data_dir=None):
    """
    Tạo file data.yaml tự động từ thư mục YOLO.
    
    Args:
        input_dir (str): Thư mục chứa classes.txt
        output_yaml (str): Đường dẫn để lưu file yaml
        data_dir (str, optional): Thư mục dữ liệu (nếu khác input_dir)
    
    Returns:
        str: Đường dẫn đến file yaml đã tạo
    """
    print(f"Tạo file data.yaml từ thư mục: {input_dir}")
    
    # Đọc classes.txt
    class_file = os.path.join(input_dir, "classes.txt")
    try:
        with open(class_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file classes.txt tại: {class_file}")
    
    # Tạo cấu trúc yaml
    yaml_content = {
        "path": os.path.abspath(data_dir or input_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names
    }
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(os.path.abspath(output_yaml)), exist_ok=True)
    
    # Ghi file yaml
    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, allow_unicode=True)
    
    print(f"✅ Đã tạo file data.yaml tại: {output_yaml}")
    return output_yaml

def convert_pdf_to_image(args):
    """
    Chuyển đổi PDF sang ảnh.
    
    Args:
        args: Đối số dòng lệnh từ ArgumentParser
    """
    print("=== CHUYỂN ĐỔI PDF SANG ẢNH ===")
    
    # Kiểm tra nếu có đường dẫn PDF
    if not args.pdf_path:
        print("❌ Thiếu đường dẫn đến file PDF. Sử dụng --pdf-path để chỉ định.")
        return
    
    # Thư mục đầu ra
    output_folder = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image')
    os.makedirs(output_folder, exist_ok=True)
    
    # Import và sử dụng hàm convert_pdf_to_images
    from src.gen_data.pdf_to_image import convert_pdf_to_images
    convert_pdf_to_images(args.pdf_path, output_folder, args.format)
    
    print(f"✅ Đã chuyển đổi PDF thành ảnh trong thư mục: {output_folder}")

def prepare_data(args):
    """
    Chuẩn bị dữ liệu từ thư mục YOLO.
    
    Args:
        args: Đối số dòng lệnh từ ArgumentParser
    """
    print("=== CHUẨN BỊ DỮ LIỆU ===")
    
    # Lưu thư mục đầu vào từ tham số dòng lệnh
    input_dir = args.input_dir
    if input_dir:
        # Cập nhật cfg cho các phần khác của chương trình
        cfg.INPUT_DATA_CONFIG["input_dir"] = input_dir
        print(f"Thư mục đầu vào: {input_dir}")
        
    # CỐ ĐỊNH thư mục đầu ra luôn là 'data/'
    cfg.OUTPUT_DATA_CONFIG["output_dir"] = "data"
    print(f"Thư mục đầu ra được cố định: data/")
    
    if args.test_ratio is not None:
        cfg.OUTPUT_DATA_CONFIG["test_ratio"] = args.test_ratio
    if args.val_ratio is not None:
        cfg.OUTPUT_DATA_CONFIG["val_ratio"] = args.val_ratio
    
    # Tạo và chạy DataPreparer với thư mục đầu vào được chỉ định
    preparer = DataPreparer(input_dir=input_dir)
    preparer.prepare_dataset()

def augment_data(args):
    print("=== TĂNG CƯỜNG DỮ LIỆU ===")
    
    if args.num_images is not None:
        cfg.AUGMENTATION_CONFIG["num_images"] = args.num_images
    if args.objects_per_image is not None:
        cfg.AUGMENTATION_CONFIG["num_objects_per_image"] = args.objects_per_image
    if args.balance_ratio is not None:
        cfg.AUGMENTATION_CONFIG["balance_ratio"] = args.balance_ratio
    
    num_images = args.num_images
    objects_per_image = args.objects_per_image
    balance_ratio = args.balance_ratio
    
    output_dir = args.output_dir
    print(f"Thư mục đầu ra: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {output_dir}")
    
    try:
        class_names = cfg.get_class_names(custom_input_dir=output_dir)
    except FileNotFoundError:
        print(f"Không tìm thấy classes.txt trong {output_dir}, thử thư mục data mặc định...")
        class_names = cfg.get_class_names()
    
    augmenter = DatasetAugmenter(
        output_dir=output_dir,
        class_names=class_names,
        num_images=num_images,
        num_objects_per_image=objects_per_image,
        balance_ratio=balance_ratio
    )
    
    if augmenter.augment_dataset():
        print(f"✅ Đã tăng cường dữ liệu và lưu vào các thư mục tương ứng trong: {output_dir}")
    else:
        print(f"⚠ Không có dữ liệu nào được tăng cường do thiếu thư mục images/ hoặc labels/ trong {output_dir}")

def train_model(args):
    """
    Huấn luyện mô hình YOLO.
    
    Args:
        args: Đối số dòng lệnh từ ArgumentParser
    """
    print("=== HUẤN LUYỆN MÔ HÌNH ===")
    
    # Ghi đè cấu hình từ tham số dòng lệnh
    if args.model:
        cfg.TRAIN_CONFIG["model_path"] = args.model
    
    if args.epochs is not None:
        cfg.TRAIN_CONFIG["train_params"]["epochs"] = args.epochs
    
    if args.batch_size is not None:
        cfg.TRAIN_CONFIG["train_params"]["batch"] = args.batch_size
    
    if args.img_size is not None:
        cfg.TRAIN_CONFIG["train_params"]["imgsz"] = args.img_size
    
    if args.model_output:
        cfg.TRAIN_CONFIG["train_params"]["project"] = args.model_output
        os.makedirs(args.model_output, exist_ok=True)
    
    # Tạo file data.yaml nếu cần
    if args.yaml_path:
        yaml_path = args.yaml_path
    else:
        yaml_path = os.path.join(os.path.dirname(cfg.get_output_dir()), cfg.TRAIN_CONFIG["yaml_path"])
        
        # CHỈ TÁC VỤ TRAIN: Luôn sử dụng thư mục data/ làm thư mục dữ liệu
        data_dir = "data"
        
        # Luôn ưu tiên sử dụng file classes.txt trong thư mục data/
        data_classes_file = os.path.join("data", "classes.txt")
        if os.path.exists(data_classes_file):
            input_dir = "data"
        # Nếu không tìm thấy trong data/, thử tìm trong thư mục output
        elif args.output_dir and os.path.exists(os.path.join(args.output_dir, "classes.txt")):
            input_dir = args.output_dir
        # Cuối cùng mới dùng thư mục input
        else:
            input_dir = args.input_dir or cfg.get_input_dir()
            
        create_data_yaml(input_dir, yaml_path, data_dir)
    
    # Tạo và chạy YOLOTrainer
    trainer = YOLOTrainer()
    trainer.train_model()
    
    project_dir = os.path.join(cfg.TRAIN_CONFIG['train_params']['project'], cfg.TRAIN_CONFIG['train_params']['name'])
    print(f"✅ Đã huấn luyện mô hình và lưu vào thư mục: {project_dir}")

def predict(args):
    """
    Thực hiện dự đoán với mô hình YOLO.
    
    Args:
        args: Đối số dòng lệnh từ ArgumentParser
    """
    print("=== DỰ ĐOÁN ===")
    
    # Ghi đè cấu hình từ tham số dòng lệnh
    if args.batch_size is not None:
        cfg.PREDICT_CONFIG["predict_params"]["batch"] = args.batch_size
    
    if args.img_size is not None:
        cfg.PREDICT_CONFIG["predict_params"]["imgsz"] = args.img_size
    
    if args.save_txt is not None:
        cfg.PREDICT_CONFIG["predict_params"]["save_txt"] = args.save_txt
    
    if args.save_conf is not None:
        cfg.PREDICT_CONFIG["predict_params"]["save_conf"] = args.save_conf
    
    # Kiểm tra xem có sử dụng best.pt hoặc đường dẫn model tùy chỉnh không
    model_path = None
    if args.best_model:
        # Kiểm tra xem args.best_model có phải là đường dẫn không
        if isinstance(args.best_model, str) and args.best_model != 'True':
            # Sử dụng đường dẫn model được chỉ định trực tiếp
            if os.path.exists(args.best_model):
                model_path = args.best_model
                print(f"Sử dụng mô hình tùy chỉnh tại: {model_path}")
            else:
                raise FileNotFoundError(f"Không tìm thấy file model tại: {args.best_model}")
        else:
            # Tìm file best.pt trong thư mục project_results
            model_dir = os.path.dirname(os.path.join(cfg.PROJECT_ROOT, cfg.TRAIN_CONFIG["model_path"]))
            if os.path.isdir(model_dir):
                for root, dirs, files in os.walk(model_dir):
                    if "weights" in dirs:
                        best_model_path = os.path.join(root, "weights", "best.pt")
                        if os.path.exists(best_model_path):
                            model_path = best_model_path
                            print(f"Sử dụng mô hình best.pt tại: {best_model_path}")
                            break
                
                if model_path is None:
                    print("Không tìm thấy file best.pt, sử dụng mô hình mặc định.")
    elif args.model:
        model_path = args.model
    
    # Tạo và chạy YOLOPredictor
    predictor = YOLOPredictor(model_path=model_path)
    
    # Validate nếu được yêu cầu
    if args.do_validate:
        predictor.validate()
    
    # Dự đoán
    source = args.source or None
    predictor.predict(source=source)
    
    print("✅ Đã hoàn thành dự đoán.")

def evaluate_model(args):
    """
    Đánh giá mô hình YOLO.
    
    Args:
        args: Đối số dòng lệnh từ ArgumentParser
    """
    print("=== ĐÁNH GIÁ MÔ HÌNH ===")
    
    # Import YOLOEvaluator
    from src.eval.eval_yolo import YOLOEvaluator
    
    # Tìm đường dẫn đến mô hình
    model_path = args.model or cfg.TRAIN_CONFIG["model_path"]
    if args.best_model:
        # Kiểm tra xem args.best_model có phải là đường dẫn không
        if isinstance(args.best_model, str) and args.best_model != 'True':
            # Sử dụng đường dẫn model được chỉ định trực tiếp
            if os.path.exists(args.best_model):
                model_path = args.best_model
                print(f"Sử dụng mô hình tùy chỉnh tại: {model_path}")
            else:
                raise FileNotFoundError(f"Không tìm thấy file model tại: {args.best_model}")
        else:
            # Tìm file best.pt trong thư mục project_results
            model_dir = os.path.dirname(os.path.join(cfg.PROJECT_ROOT, model_path))
            if os.path.isdir(model_dir):
                for root, dirs, files in os.walk(model_dir):
                    if "weights" in dirs:
                        best_model_path = os.path.join(root, "weights", "best.pt")
                        if os.path.exists(best_model_path):
                            model_path = best_model_path
                            print(f"Sử dụng mô hình best.pt tại: {best_model_path}")
                            break
                
                if model_path == args.model or model_path == cfg.TRAIN_CONFIG["model_path"]:
                    print("Không tìm thấy file best.pt, sử dụng mô hình mặc định.")
    
    # Tìm file YAML
    yaml_path = args.yaml_path or cfg.TRAIN_CONFIG["yaml_path"]
    
    # Tìm tên các lớp
    try:
        class_names = cfg.get_class_names()
    except:
        class_names = []
        # Đọc từ file classes.txt
        class_file = os.path.join(cfg.get_input_dir(), "classes.txt")
        if os.path.exists(class_file):
            with open(class_file, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f if line.strip()]
    
    # Cấu hình đánh giá
    eval_config = {
        "model_path": model_path,
        "yaml_path": yaml_path,
        "class_names": class_names,
        "eval_params": {
            "imgsz": args.img_size or cfg.PREDICT_CONFIG["predict_params"]["imgsz"],
            "batch": args.batch_size or cfg.PREDICT_CONFIG["predict_params"]["batch"]
        }
    }
    
    # Tạo evaluator
    evaluator = YOLOEvaluator(eval_config)
    
    # Đánh giá trên các tập dữ liệu được yêu cầu
    splits = args.split.split(",") if args.split else ["test"]
    results = {}
    
    for split in splits:
        print(f"\nĐánh giá trên tập {split}:")
        metrics = evaluator.evaluate(split=split)
        evaluator.print_metrics(split, metrics)
        results[split] = metrics
    
    # Lưu kết quả nếu được yêu cầu
    if args.save_metrics:
        import json
        metrics_dir = args.metrics_output or "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        
        for split, metrics in results.items():
            # Chuyển đổi các mảng NumPy thành danh sách Python
            serializable_metrics = {}
            for key, value in metrics.items():
                if hasattr(value, "tolist"):
                    serializable_metrics[key] = value.tolist()
                else:
                    serializable_metrics[key] = value
            
            metrics_file = os.path.join(metrics_dir, f"{split}_metrics.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, indent=4)
            print(f"Đã lưu kết quả đánh giá tập {split} vào file: {metrics_file}")
    
    print("✅ Đã hoàn thành đánh giá mô hình.")

def main():
    parser = argparse.ArgumentParser(description="Công cụ huấn luyện và dự đoán YOLO")
    parser.add_argument('--task', type=str, required=True, 
                       choices=['prepare', 'augment', 'train', 'predict', 'pdf_to_image', 'eval', 'all'],
                       help='Nhiệm vụ cần thực hiện: prepare (chuẩn bị dữ liệu), augment (tăng cường dữ liệu), ' +
                            'train (huấn luyện), predict (dự đoán), pdf_to_image (chuyển PDF sang ảnh), ' +
                            'eval (đánh giá mô hình), all (thực hiện tất cả)')
    
    # Các đối số chung
    parser.add_argument('--input-dir', type=str, help='Thư mục dữ liệu YOLO đầu vào (ghi đè cfg.INPUT_DATA_CONFIG["input_dir"])')
    parser.add_argument('--output-dir', type=str, help='Thư mục đầu ra (ghi đè cfg.OUTPUT_DATA_CONFIG["output_dir"])')
    
    # Các đối số cho PDF to Image
    parser.add_argument('--pdf-path', type=str, help='Đường dẫn đến file PDF cần chuyển đổi')
    parser.add_argument('--format', type=str, default='JPEG', choices=['JPEG', 'PNG', 'BMP'], help='Định dạng ảnh đầu ra')
    
    # Các đối số cho Prepare Data
    parser.add_argument('--test-ratio', type=float, help='Tỉ lệ dữ liệu test (ghi đè cfg.OUTPUT_DATA_CONFIG["test_ratio"])')
    parser.add_argument('--val-ratio', type=float, help='Tỉ lệ dữ liệu validation (ghi đè cfg.OUTPUT_DATA_CONFIG["val_ratio"])')
    
    # Các đối số cho Augmentation
    parser.add_argument('--num-images', type=int, help='Số lượng ảnh cần tạo (ghi đè cfg.AUGMENTATION_CONFIG["num_images"])')
    parser.add_argument('--objects-per-image', type=int, help='Số đối tượng trên mỗi ảnh (ghi đè cfg.AUGMENTATION_CONFIG["num_objects_per_image"])')
    parser.add_argument('--balance-ratio', type=float, help='Tỉ lệ cân bằng (ghi đè cfg.AUGMENTATION_CONFIG["balance_ratio"])')
    parser.add_argument('--augment-input', type=str, help='Thư mục ảnh đầu vào cho augmentation')
    parser.add_argument('--augment-input-labels', type=str, help='Thư mục nhãn đầu vào cho augmentation')
    parser.add_argument('--augment-output', type=str, help='Thư mục đầu ra cho augmentation')
    
    # Các đối số cho Training
    parser.add_argument('--model', type=str, help='Đường dẫn đến mô hình pretrained (ghi đè cfg.TRAIN_CONFIG["model_path"])')
    parser.add_argument('--epochs', type=int, help='Số epochs (ghi đè cfg.TRAIN_CONFIG["train_params"]["epochs"])')
    parser.add_argument('--batch-size', type=int, help='Kích thước batch')
    parser.add_argument('--img-size', type=int, help='Kích thước ảnh đầu vào')
    parser.add_argument('--model-output', type=str, help='Thư mục lưu mô hình đầu ra')
    parser.add_argument('--yaml-path', type=str, help='Đường dẫn đến file YAML')
    
    # Các đối số cho Prediction
    parser.add_argument('--source', type=str, help='Nguồn dữ liệu để dự đoán (ảnh, thư mục ảnh, video)')
    parser.add_argument('--save-txt', action='store_true', help='Lưu kết quả dưới dạng file txt')
    parser.add_argument('--save-conf', action='store_true', help='Lưu độ tin cậy trong file txt')
    parser.add_argument('--do-validate', action='store_true', help='Thực hiện validate trước khi dự đoán')
    
    # Các đối số cho Evaluation
    parser.add_argument('--split', type=str, help='Tập dữ liệu để đánh giá (train,val,test hoặc riêng lẻ)')
    parser.add_argument('--save-metrics', action='store_true', help='Lưu kết quả đánh giá vào file')
    parser.add_argument('--metrics-output', type=str, help='Thư mục lưu kết quả đánh giá')
    
    # Đối số chung cho predict và eval
    parser.add_argument('--best-model', nargs='?', const=True, 
                      help='Sử dụng file best.pt trong thư mục weights hoặc đường dẫn tùy chỉnh đến model. '
                           'Nếu sử dụng không có giá trị (--best-model), sẽ tự động tìm best.pt. '
                           'Nếu truyền đường dẫn (--best-model /đường/dẫn/model.pt), sẽ sử dụng model đó.')
    
    args = parser.parse_args()
    
    try:
        if args.task == 'prepare':
            prepare_data(args)
            
        elif args.task == 'pdf_to_image':
            convert_pdf_to_image(args)
            
        elif args.task == 'augment':
            augment_data(args)
            
        elif args.task == 'train':
            train_model(args)
            
        elif args.task == 'predict':
            predict(args)
            
        elif args.task == 'eval':
            evaluate_model(args)
            
        elif args.task == 'all':
            # Thực hiện tất cả các bước
            prepare_data(args)
            print("")
            augment_data(args)
            print("")
            train_model(args)
            print("")
            predict(args)
            print("")
            evaluate_model(args)
    
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
