import os
import subprocess
import sys
import time

def run_step(module_name, step_name):
    """Hàm chạy một module độc lập và đo lường thời gian"""
    print(f"\n{'='*70}")
    print(f"BẮT ĐẦU BƯỚC: {step_name}".center(70))
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Dùng subprocess để chạy script. sys.executable đảm bảo luôn dùng đúng Python của môi trường ảo (venv)
    result = subprocess.run([sys.executable, "-m", module_name])
    
    elapsed_time = time.time() - start_time
    
    # Kiểm tra mã lỗi (0 là thành công, khác 0 là có lỗi)
    if result.returncode != 0:
        print(f"\n LỖI NGHIÊM TRỌNG: Quá trình '{step_name}' thất bại tại module '{module_name}'.")
        print(" Pipeline đã bị buộc dừng để bảo vệ dữ liệu.")
        sys.exit(1)
        
    print(f"\n KẾT THÚC: {step_name} (Thời gian thực thi: {elapsed_time:.2f} giây)")

def main():
    print("\n" + "#"*70)
    print(" HỆ THỐNG PIPELINE TỰ ĐỘNG - DỰ ĐOÁN DOANH THU BÁN LẺ ".center(70))
    print("#"*70)
    
    total_start = time.time()
    
    # Định nghĩa luồng chạy chuẩn: Preprocess -> Train -> Evaluate
    pipeline_steps = [
        ("Tiền xử lý & Trích xuất đặc trưng (Preprocess)", "src.preprocess"),
        ("Huấn luyện 4 Mô hình (Train from scratch)", "src.train"),
        ("Đánh giá sai số & Báo cáo (Evaluate)", "src.evaluate")
    ]
    
    # Thực thi tuần tự
    for step_name, module_name in pipeline_steps:
        run_step(module_name, step_name)
        
    total_elapsed = time.time() - total_start
    
    print("\n" + "#"*70)
    print(f" PIPELINE CHẠY THÀNH CÔNG TOÀN BỘ (Tổng thời gian: {total_elapsed:.2f}s) 🎉".center(70))
    print(" Bạn đã có thể dùng các file .pkl trong models/ để mang đi Deploy! ".center(70))
    print("#"*70 + "\n")

if __name__ == "__main__":
    main()