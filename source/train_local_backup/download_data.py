import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_video(row):
    video_id = row['Id']
    url = row['Download_link']
    # Lấy thư mục đích từ dòng dữ liệu (được gán ở hàm main)
    save_folder = row['target_folder'] 
    
    # Tạo tên file: ID.mp4
    file_path = os.path.join(save_folder, f"{video_id}.mp4")
    
    # Nếu file đã tồn tại thì bỏ qua
    if os.path.exists(file_path):
        return "Existed"
    
    try:
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                    if chunk:
                        f.write(chunk)
            return "Success"
        else:
            return f"Failed: Status {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # 1. Định nghĩa đường dẫn
    train_dir = "./data/train_videos"
    val_dir = "./data/val_videos"

    # Tạo thư mục nếu chưa có
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
        
    # 2. Đọc file CSV
    print("Reading CSV files...")
    df_train = pd.read_csv("./data/train_data.csv")
    df_val = pd.read_csv("./data/val_data.csv")
    
    # --- ĐIỂM KHÁC BIỆT: Gán thư mục đích cho từng DataFrame ---
    # Thêm cột 'target_folder' để hàm download biết lưu vào đâu
    df_train['target_folder'] = train_dir
    df_val['target_folder'] = val_dir
    
    # Gộp lại để chạy chung 1 pool thread cho tối ưu băng thông
    # (Vẫn giữ nguyên logic tách folder nhờ cột target_folder)
    df_all = pd.concat([
        df_train[['Id', 'Download_link', 'target_folder']], 
        df_val[['Id', 'Download_link', 'target_folder']]
    ])
    
    # Loại bỏ trùng lặp (nếu có ID trùng trong cùng 1 tập)
    # Lưu ý: Nếu 1 ID xuất hiện cả ở Train và Val (hiếm), nó sẽ được giữ cả 2 dòng 
    # và tải vào cả 2 folder (điều này đúng với ý định tách biệt của bạn).
    df_all = df_all.drop_duplicates(subset=['Id', 'target_folder'])
    
    print(f"Total videos to download: {len(df_all)}")
    print(f" - Train: {len(df_train)}")
    print(f" - Val: {len(df_val)}")
    
    # 3. Tải đa luồng
    rows = [row for _, row in df_all.iterrows()]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(download_video, rows), total=len(rows), unit="vid"))
        
    # Thống kê
    success_count = results.count("Success")
    existed_count = results.count("Existed")
    fail_count = len(results) - success_count - existed_count
    
    print("\n--- Download Summary ---")
    print(f"Success: {success_count}")
    print(f"Already Existed: {existed_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    main()
