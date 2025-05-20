'''
Sử dụng để chuyển đổi file pdf sang ảnh thuận tiện cho việc Labeling bằng Label Studio
'''

from pdf2image import convert_from_path
import os
import argparse
# Đường dẫn đến file PDF
def convert_pdf_to_images(pdf_path: str, output_folder: str, image_format: str = "JPEG") -> None:
    """
    Chuyển mỗi trang của file PDF thành một ảnh và lưu vào thư mục chỉ định.

    Args:
        pdf_path (str): Đường dẫn file PDF.
        output_folder (str): Thư mục đầu ra để lưu ảnh.
        image_format (str): Định dạng ảnh đầu ra, mặc định là "JPEG".

    Returns:
        None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        output_path = os.path.join(output_folder, f"page_{i+1}.{image_format.lower()}")
        image.save(output_path, image_format)
        print(f"Đã lưu ảnh: {output_path}")
    print(f"Tất cả các trang đã được chuyển đổi và lưu vào {output_folder}")

def parse_arguments():
    """Định nghĩa và phân tích các tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description="Chuyển đổi file PDF thành ảnh.")
    parser.add_argument("pdf_path", type=str, help="Đường dẫn đến file PDF cần chuyển đổi.")
    parser.add_argument("--output_folder", type=str, default="images", help="Thư mục đầu ra để lưu ảnh (mặc định là 'images').")
    parser.add_argument("--image_format", type=str, default="JPEG", choices=["JPEG", "PNG", "BMP"], help="Định dạng ảnh đầu ra (mặc định là 'JPEG').")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    convert_pdf_to_images(args.pdf_path, args.output_folder, args.image_format)
