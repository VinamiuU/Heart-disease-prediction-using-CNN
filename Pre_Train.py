import pandas as pd
import numpy as np
import wfdb
import ast
import os
from sklearn.preprocessing import MultiLabelBinarizer # Để mã hóa nhãn
from sklearn.model_selection import train_test_split # Để chia tập validation

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def load_raw_data(df_metadata_subset, current_sampling_rate, base_path):
    """
    Tải dữ liệu tín hiệu ECG thô từ các tệp, sử dụng kiểu dữ liệu float32.
    Trả về mảng tín hiệu và danh sách các ecg_id đã tải thành công.
    """
    signals_list = []
    filenames_to_process = []
    original_indices = [] 

    if current_sampling_rate == 100:
        filenames_to_process = df_metadata_subset.filename_lr
        original_indices = df_metadata_subset.index
    elif current_sampling_rate == 500:
        filenames_to_process = df_metadata_subset.filename_hr
        original_indices = df_metadata_subset.index
    else:
        # Giữ lại kiểm tra lỗi này vì nó quan trọng cho logic của hàm
        raise ValueError(f"Tần số lấy mẫu không được hỗ trợ: {current_sampling_rate}")

    print(f"   - Chuẩn bị tải {len(filenames_to_process)} bản ghi ở {current_sampling_rate}Hz...")
    
    successful_signals = []
    successful_indices = [] 

    for idx, f_rel_path in zip(original_indices, filenames_to_process):
        if pd.isna(f_rel_path) or f_rel_path.strip() == "":
            continue # Bỏ qua file nếu tên file không hợp lệ

        full_file_path = os.path.join(base_path, f_rel_path)
        try:
            signal_data, meta = wfdb.rdsamp(full_file_path)
            successful_signals.append(signal_data.astype(np.float32)) 
            successful_indices.append(idx) 
        except Exception:
            # Lỗi tải file cụ thể sẽ được bỏ qua, không in ra để làm gọn
            pass 
        
    print(f"   - Hoàn tất tải. Tổng số tín hiệu tải thành công: {len(successful_signals)}.")
    
    if not successful_signals:
        return np.array([], dtype=np.float32), []

    try:
        final_signals_array = np.array(successful_signals, dtype=np.float32)
        return final_signals_array, successful_indices
    except ValueError as ve: 
        # Giữ lại thông báo lỗi này vì nó quan trọng nếu mảng không tạo được
        print(f"Lỗi nghiêm trọng khi tạo mảng NumPy từ tín hiệu: {ve}.")
        return successful_signals, successful_indices


# --- 0. Cấu hình ---
print("---0: CẤU HÌNH ---")
PATH_TO_DATASET = r'D:/Learning/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1'
SAMPLING_RATE = 100 # Hz
MAX_RECORDS_TO_LOAD = 17000 
TEST_FOLD_NUMBER = 10 
VALIDATION_SPLIT_SIZE = 0.1 
RANDOM_STATE = 42 

# --- 1. Tải và Chuyển đổi Metadata ---
#print("\n---1: Load ptbxl_database.csv ---")
database_csv_path = os.path.join(PATH_TO_DATASET, 'ptbxl_database.csv')
Y_metadata = pd.read_csv(database_csv_path, index_col='ecg_id')
Y_metadata.scp_codes = Y_metadata.scp_codes.apply(lambda x: ast.literal_eval(x))
print(f"DONE. Số lượng {len(Y_metadata)} record: ")

# --- 2. Lọc Metadata và Chọn Số Lượng Bản Ghi ---
#print(f"\n---2: Lọc METADAta VÀ CHỌN SỐ LƯỢNG BẢN GHI (Rate: {SAMPLING_RATE}Hz) ---")
filename_column = 'filename_lr' if SAMPLING_RATE == 100 else 'filename_hr'
if filename_column not in Y_metadata.columns:
    # Giữ lại kiểm tra lỗi này
    raise KeyError(f"Thiếu cột '{filename_column}' trong ptbxl_database.csv cho sampling_rate={SAMPLING_RATE}")

valid_filenames_mask = Y_metadata[filename_column].notna() & (Y_metadata[filename_column].str.strip() != '')
Y_initially_filtered = Y_metadata[valid_filenames_mask].copy()
print(f"   Số bản ghi có tên file hợp lệ: {len(Y_initially_filtered)}")

Y_for_loading = Y_initially_filtered
if MAX_RECORDS_TO_LOAD is not None and MAX_RECORDS_TO_LOAD > 0:
    if MAX_RECORDS_TO_LOAD < len(Y_initially_filtered):
        print(f"   Giới hạn số lượng bản ghi xuống còn {MAX_RECORDS_TO_LOAD} (lấy mẫu ngẫu nhiên).")
        Y_for_loading = Y_initially_filtered.sample(n=MAX_RECORDS_TO_LOAD, random_state=RANDOM_STATE)
    # Không cần print cho trường hợp MAX_RECORDS_TO_LOAD lớn hơn hoặc bằng
elif MAX_RECORDS_TO_LOAD is None:
    print(f"   Sẽ cố gắng tải tất cả {len(Y_initially_filtered)} bản ghi hợp lệ.")

# --- 3. Tải Dữ liệu Tín hiệu Thô ---
#print(f"\n--- 3: call load_raw_data ---")
X_signals, successful_ecg_ids = [], [] # Khởi tạo
if not Y_for_loading.empty:
    X_signals, successful_ecg_ids = load_raw_data(Y_for_loading, SAMPLING_RATE, PATH_TO_DATASET)
else:
    print("   Không có bản ghi nào để tải tín hiệu.")
    X_signals = np.array([], dtype=np.float32)

Y_final_metadata = pd.DataFrame(columns=Y_for_loading.columns) # Khởi tạo DataFrame rỗng
if successful_ecg_ids: # Chỉ thực hiện nếu có tín hiệu được tải thành công
    Y_final_metadata = Y_for_loading.loc[successful_ecg_ids].copy()
    print(f"   Đồng bộ hóa: Y_final_metadata chứa {len(Y_final_metadata)} bản ghi, X_signals chứa {X_signals.shape[0] if isinstance(X_signals, np.ndarray) else len(X_signals)} tín hiệu.")
else: 
    if isinstance(X_signals, list) and not X_signals: # Đảm bảo X_signals là mảng numpy rỗng nếu không có gì được tải
         X_signals = np.array([], dtype=np.float32)

# --- 4. Xử lý Nhãn Chẩn Đoán và Chia Dữ Liệu ---
# Chỉ tiếp tục nếu có dữ liệu và X_signals, Y_final_metadata đồng bộ
if (isinstance(X_signals, np.ndarray) and X_signals.shape[0] > 0) and \
   not Y_final_metadata.empty and \
   (X_signals.shape[0] == len(Y_final_metadata)):

    print(f"\n--- 4: XỬ LÝ NHÃN VÀ CHIA TẬP DỮ LIỆU ---")
    statements_csv_path = os.path.join(PATH_TO_DATASET, 'scp_statements.csv')
    agg_df = pd.read_csv(statements_csv_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp)) 

    Y_final_metadata['diagnostic_superclass'] = Y_final_metadata.scp_codes.apply(aggregate_diagnostic)
    print("   - Tổng hợp siêu lớp chẩn đoán hoàn tất.")

    print(f"   - Chia dữ liệu dựa trên test_fold = {TEST_FOLD_NUMBER}...")
    train_val_mask = (Y_final_metadata.strat_fold != TEST_FOLD_NUMBER)
    test_mask = (Y_final_metadata.strat_fold == TEST_FOLD_NUMBER)

    X_train_val = X_signals[train_val_mask.values]
    y_train_val_labels = Y_final_metadata.loc[train_val_mask, 'diagnostic_superclass']
    
    X_test = X_signals[test_mask.values]
    y_test_labels = Y_final_metadata.loc[test_mask, 'diagnostic_superclass']
    print(f"     Kích thước X_train_val ban đầu: {X_train_val.shape}, y_train_val: {y_train_val_labels.shape}")
    print(f"     Kích thước X_test: {X_test.shape}, y_test: {y_test_labels.shape}")

    # --- 5. Tiền xử lý Nhãn (Multi-hot encoding) ---
    print(f"\n---  5: TIỀN XỬ LÝ NHÃN ---")
    temp_mlb = MultiLabelBinarizer()
    # Fit trên toàn bộ nhãn đã tải để có danh sách lớp đầy đủ
    all_labels_for_mlb_fit = Y_final_metadata['diagnostic_superclass']
    temp_mlb.fit(all_labels_for_mlb_fit) 
    all_classes = list(temp_mlb.classes_)
    print(f"   - Các siêu lớp chẩn đoán tìm thấy: {all_classes}")

    mlb = MultiLabelBinarizer(classes=all_classes) 
    y_train_val_encoded = mlb.fit_transform(y_train_val_labels)
    y_test_encoded = mlb.transform(y_test_labels) 
    print(f"   - Kích thước y_train_val_encoded: {y_train_val_encoded.shape}")
    print(f"   - Kích thước y_test_encoded: {y_test_encoded.shape}")

    # --- 6. Chia tập Huấn luyện và Tập Xác thực từ X_train_val ---
    print(f"\n---  6: CHIA TẬP HUẤN LUYỆN VÀ TẬP XÁC THỰC ---")
    X_train, X_val, y_train, y_val = [np.array([])]*4 # Khởi tạo rỗng

    if len(X_train_val) > 1 and VALIDATION_SPLIT_SIZE > 0 :
        try:
            # Kiểm tra xem có thể stratify không
            can_stratify = np.sum(y_train_val_encoded.sum(axis=0) > 1) > 0 and y_train_val_encoded.shape[0] > 1
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, 
                y_train_val_encoded, 
                test_size=VALIDATION_SPLIT_SIZE, 
                random_state=RANDOM_STATE,
                stratify=y_train_val_encoded if can_stratify else None
            )
            print(f"   - Đã chia với stratify={'có' if can_stratify else 'không'}.")
        except ValueError: # Xử lý lỗi nếu stratify không thành công
            print(f"   - Lỗi khi chia với stratify. Thử lại không có stratify.")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, 
                y_train_val_encoded, 
                test_size=VALIDATION_SPLIT_SIZE, 
                random_state=RANDOM_STATE,
                stratify=None 
            )
            print(f"   - Đã chia mà không có stratify.")
            
        print(f" - Kích thước X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f" - Kích thước X_val: {X_val.shape}, y_val: {y_val.shape}")

    elif len(X_train_val) > 0 : 
        print("   - Không chia tập xác thực, X_train sẽ là toàn bộ X_train_val.")
        X_train, y_train = X_train_val, y_train_val_encoded
    else: 
        print("   - X_train_val rỗng, không thể chia.")

    # --- 7. Tiền xử lý Tín hiệu X (Standardization) ---
    print(f"\n---  7: TIỀN XỬ LÝ TÍN HIỆU X (STANDARDIZATION) ---")
    if X_train.size > 0:
        num_samples_train, num_timesteps, num_features = X_train.shape
        # Reshape để tính mean/std cho từng kênh trên toàn bộ dữ liệu huấn luyện
        X_train_reshaped = X_train.reshape(-1, num_features)
        
        X_train_channel_mean = np.mean(X_train_reshaped, axis=0)
        X_train_channel_std = np.std(X_train_reshaped, axis=0)
        # Tránh chia cho 0 nếu std của kênh nào đó bằng 0
        X_train_channel_std[X_train_channel_std == 0] = 1e-9 

        # Áp dụng chuẩn hóa
        X_train = (X_train - X_train_channel_mean) / X_train_channel_std
        if X_val.size > 0:
            X_val = (X_val - X_train_channel_mean) / X_train_channel_std 
        if X_test.size > 0: 
            X_test = (X_test - X_train_channel_mean) / X_train_channel_std
        
        print(f"   - Chuẩn hóa Z-score cho X_train, X_val, X_test hoàn tất (dựa trên X_train).")
    else:
        print("   - Không có dữ liệu X_train để chuẩn hóa.")

else:
    print("\nKhông có đủ dữ liệu X_signals hoặc Y_final_metadata để xử lý thêm. Kết thúc.")
    # Khởi tạo các biến cuối cùng thành mảng rỗng để tránh lỗi nếu truy cập sau này
    X_train, y_train, X_val, y_val, X_test, y_test_encoded = [np.array([])]*6


print("\n--- SCRIPT TIỀN XỬ LÝ HOÀN TẤT ---")
# Kiểm tra cuối cùng trước khi in thông tin các biến sẵn sàng
data_available_for_training = (isinstance(X_signals, np.ndarray) and X_signals.shape[0] > 0) and \
                              not Y_final_metadata.empty and \
                              (X_signals.shape[0] == len(Y_final_metadata))

if data_available_for_training:
    print("Các biến sau đã sẵn sàng cho việc huấn luyện mô hình:")
    print(f"   X_train: {X_train.shape if X_train.size > 0 else 'Rỗng'}")
    print(f"   y_train (đã mã hóa): {y_train.shape if y_train.size > 0 else 'Rỗng'}")
    print(f"   X_val: {X_val.shape if X_val.size > 0 else 'Rỗng'}")
    print(f"   y_val (đã mã hóa): {y_val.shape if y_val.size > 0 else 'Rỗng'}")
    print(f"   X_test: {X_test.shape if X_test.size > 0 else 'Rỗng'}")
    print(f"   y_test_encoded (đã mã hóa): {y_test_encoded.shape if hasattr(y_test_encoded, 'shape') and y_test_encoded.size > 0 else 'Rỗng'}")
    print(f"   mlb (MultiLabelBinarizer đã fit): {'Có' if 'mlb' in locals() and hasattr(mlb, 'classes_') else 'Không'}")
else:
    print("Không có dữ liệu cuối cùng cho các biến train/test/val do lỗi hoặc không có dữ liệu đầu vào.")

print("\n--- SCRIPT TRAIN MODEL---\n")

if 'X_train' not in locals() or X_train.size == 0:
    print("No X train.")

# --- Định nghĩa các tham số cho mô hình ---
input_shape = (X_train.shape[1], X_train.shape[2]) # (số_bước_thời_gian, số_kênh/chuyển_đạo)
num_classes = y_train.shape[1] # Số lượng lớp duy nhất (từ y_train đã mã hóa multi-hot)

learning_rate = 0.001
batch_size = 32 
epochs = 25   

print(f"--- THÔNG SỐ MÔ HÌNH ---")
print(f"Input shape: {input_shape}")
print(f"Số lượng lớp (num_classes): {num_classes}")
print(f"Learning rate: {learning_rate}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {epochs}")

# --- BƯỚC 8: XÂY DỰNG KIẾN TRÚC MÔ HÌNH ---
print("\n--- 8: XÂY DỰNG KIẾN TRÚC MÔ HÌNH ---")
model = Sequential([
    Conv1D(filters=64, kernel_size=7, activation='relu', padding='same', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Flatten(), 

    Dense(units=256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5), 

    Dense(units=num_classes, activation='sigmoid')
])
model.summary() 

# --- BƯỚC 9: BIÊN DỊCH MÔ HÌNH ---
print("\n---  9: BIÊN DỊCH MÔ HÌNH ---") 
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc', multi_label=True, num_labels=num_classes if num_classes > 0 else None)
    ]
)
print("Biên dịch mô hình hoàn tất.")

# --- BƯỚC 10: ĐỊNH NGHĨA CALLBACKS ---
print("\n---  10:CALLBACKS ---")  
checkpoint_filepath = 'best_ecg_model.keras' 
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=False, 
    monitor='val_auc', mode='max', save_best_only=True)     

early_stopping_callback = EarlyStopping(
    monitor='val_auc', patience=10, mode='max', restore_best_weights=True) 

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)     

callbacks_list = [model_checkpoint_callback, early_stopping_callback, reduce_lr_callback]



# --- 11 Huấn luyện Mô hình ---
# Chỉ huấn luyện nếu có dữ liệu X_train và X_val
if X_train.size > 0 and X_val.size > 0:
    print("\n--- BƯỚC 11: HUẤN LUYỆN MÔ HÌNH ---")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1) # verbose=1 để xem tiến trình huấn luyện
    print("Train Done.")
else:
    print("\nKhông có đủ dữ liệu X_train hoặc X_val để huấn luyện. Bỏ qua bước huấn luyện.")

# --- BƯỚC 12: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ---
print("\n--- BƯỚC 12: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ---")
# Kiểm tra xem model có tồn tại và có thể đánh giá không
if 'model' in locals() and hasattr(model, 'evaluate'):
    # Kiểm tra dữ liệu test
    if X_test.size > 0 and hasattr(y_test_encoded, 'shape') and y_test_encoded.size > 0:
        # Kiểm tra kes tương thích input shape
        if hasattr(model, 'input_shape') and model.input_shape is not None and X_test.shape[-1] == model.input_shape[-1]:
            
            results = model.evaluate(X_test, y_test_encoded, batch_size=batch_size, verbose=0)
            print("\nKết quả trên tập test:")
            results_list = results if isinstance(results, (list, tuple)) else [results]
            
            # Danh sách tên metrics dự kiến dựa trên thứ tự compile: loss, binary_accuracy, precision, recall, auc
            expected_names = ['loss', 'binary_accuracy', 'precision', 'recall', 'auc'] 

            if len(results_list) == len(expected_names):
                # Nếu số lượng kết quả khớp với số lượng tên dự kiến (5)
                # Ưu tiên sử dụng model.metrics_names NẾU nó cũng có 5 phần tử
                if hasattr(model, 'metrics_names') and len(model.metrics_names) == len(expected_names):
                    print("   (Sử dụng tên từ model.metrics_names)")
                    current_names_to_use = model.metrics_names
                else:
                    if hasattr(model, 'metrics_names'):
                            print(f" ")
                    else:
                            print(f"   (Sử dụng tên dự kiến do model không có thuộc tính metrics_names hoặc nó rỗng)")
                    current_names_to_use = expected_names
                
                results_dict = dict(zip(current_names_to_use, results_list))
                for name, value in results_dict.items():
                    print(f"      {name}: {value:.4f}")
            elif results_list: 
                print(f"   ⚠️ Số lượng kết quả ({len(results_list)}) không khớp với số lượng tên dự kiến ({len(expected_names)}).")
                if hasattr(model, 'metrics_names'):
                    print(f"      model.metrics_names: {model.metrics_names}")
                else:
                    print(f"      model không có thuộc tính metrics_names.")
                print(f"      Giá trị results thô:")
                for i, res_val in enumerate(results_list):
                    name_to_print = model.metrics_names[i] if hasattr(model, 'metrics_names') and i < len(model.metrics_names) else f"Value_{i+1}"
                    print(f"         {name_to_print}: {res_val:.4f}")
            else:
                print("   ⚠️ Không có kết quả đánh giá để hiển thị.")
        elif hasattr(model, 'input_shape') and model.input_shape is not None :
                print(f"\n   ⚠️ Không thể đánh giá: Số features của X_test ({X_test.shape[-1]}) không khớp với input của model ({model.input_shape[-1]}).")
        else:
                print(f"\n   ⚠️ Không thể đánh giá: Model không có thông tin input_shape.")
    else:
        print("\n   ⚠️ Dữ liệu test (X_test, y_test_encoded) không hợp lệ để đánh giá.")
else:
    print("\n   ⚠️ Mô hình chưa được huấn luyện hoặc không tồn tại để đánh giá.")

print("\n---THE END ---")