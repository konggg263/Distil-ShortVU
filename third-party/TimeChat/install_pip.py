import yaml
import subprocess
import sys

# Đổi tên file dưới đây thành tên file yml của bạn
filename = 'third-party/TimeChat/environment_macarm.yml' 

try:
    with open(filename, 'r') as f:
        env_data = yaml.safe_load(f)
        
    dependencies = env_data.get('dependencies', [])
    
    pip_packages = []
    
    for dep in dependencies:
        # Trường hợp 1: Gói tin dạng chuỗi (thường là gói conda, nhưng pip có thể thử cài)
        if isinstance(dep, str):
            # Bỏ qua python version hoặc các gói chỉ có bên conda nếu cần
            if not dep.startswith('python='): 
                pip_packages.append(dep)
        
        # Trường hợp 2: Gói tin nằm trong mục 'pip' riêng biệt
        elif isinstance(dep, dict) and 'pip' in dep:
            pip_packages.extend(dep['pip'])

    print(f"Tìm thấy {len(pip_packages)} gói tin. Đang bắt đầu cài đặt...")
    
    # Chạy lệnh pip install cho từng gói
    for package in pip_packages:
        print(f"Dang cai: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
    print("Hoàn tất!")

except FileNotFoundError:
    print(f"Không tìm thấy file {filename}")
except Exception as e:
    print(f"Có lỗi xảy ra: {e}")
