# Đề tài 5. IBM HR Analytics Employee Attrition & Performance

**Nhóm 6 · Học phần Khoa học dữ liệu trong kinh tế và kinh doanh**  
Đại học Kinh tế Quốc dân · Lớp AEP.TOKT1138.LS(126)_01

**Repository GitHub của nhóm:** https://github.com/NgocMiinh/ibm-hr-attrition-project5

## Giới thiệu

Dự án khai thác tập dữ liệu **IBM HR Analytics Employee Attrition & Performance** gồm 1.470 quan sát và 35 thuộc tính để phân tích mối liên hệ giữa tình trạng nghỉ việc với đặc điểm công việc, thâm niên, thu nhập, mức độ hài lòng và chế độ đãi ngộ.

**Lưu ý về dữ liệu:** Đây là bộ dữ liệu giả lập do các nhà khoa học dữ liệu IBM tạo ra, công bố trên Kaggle; không đại diện cho hồ sơ nhân sự thật của IBM. Tỷ lệ 16,12% là tỷ trọng nhãn Attrition = Yes trong mẫu, không phải tỷ lệ nhân viên IBM nghỉ việc hằng năm.

Nguồn: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

## Các tệp chính

| Đường dẫn | Nội dung |
| --- | --- |
| `Bao_cao_Nhom6_Project5.pdf` | Báo cáo chính thức để nộp |
| `Bao_cao_Nhom6_Project5.docx` | Phiên bản Word có thể chỉnh sửa |
| `IBM_HR_Attrition_Analysis.ipynb` | Notebook phân tích gồm các bước đọc dữ liệu, kiểm định, trực quan hóa và thử nghiệm mô hình |
| `analysis.py` | Mã Python chạy độc lập, đồng bộ với Notebook |
| `data/WA_Fn-UseC_-HR-Employee-Attrition.csv` | Tệp dữ liệu gốc |
| `figures/` | 16 biểu đồ được đặt tên theo thứ tự Hình 1 đến Hình 16 trong báo cáo |
| `hypothesis_tests.csv` | Kết quả kiểm định chín giả thuyết |
| `model_metrics.csv` | Kết quả so sánh mô hình ban đầu trên tập kiểm tra |
| `iqr_all_numeric.csv` | Kiểm tra IQR trên toàn bộ các cột số phù hợp |
| `robustness_checks.csv` | Kết quả so sánh RobustScaler và cách mã hóa biến thứ bậc |
| `analysis_results.json` | Dữ liệu tổng hợp để đối chiếu bảng và hình |
| `requirements.txt` | Danh sách thư viện cần cài đặt |

## Chạy trực tiếp từ GitHub trên Google Colab

1. Mở https://colab.research.google.com/.
2. Chọn **File → Open notebook → GitHub** và dán https://github.com/NgocMiinh/ibm-hr-attrition-project5.
3. Chọn tệp `IBM_HR_Attrition_Analysis.ipynb`.
4. Chọn **Runtime → Run all**. Khi mở Notebook riêng từ GitHub, tệp `data/` không được Colab tải tự động cùng Notebook; vì vậy chương trình sẽ thử lấy dữ liệu gốc từ Kaggle. Có thể tải CSV thủ công vào thư mục `data/` hoặc thư mục làm việc nếu mạng bị hạn chế.
5. Kiểm tra các bảng in ra và thư mục `figures` được tạo. Notebook sinh cùng các chỉ số với `analysis.py` khi thư viện và dữ liệu tương đương.

## Chạy trên máy tính

Sau khi giải nén hoặc clone repository, mở Terminal tại thư mục dự án và chạy:

```bash
python -m pip install -r requirements.txt
python analysis.py
```

Sau khi chạy, chương trình lưu kết quả vào `figures/`, `analysis_results.json`, `hypothesis_tests.csv`, `model_metrics.csv`, `iqr_all_numeric.csv` và `robustness_checks.csv`.

## Phương pháp và giới hạn

Báo cáo kiểm tra dữ liệu thiếu, trùng lặp và ràng buộc logic; dùng IQR như công cụ cảnh báo; kiểm định Chi bình phương và Mann Whitney U với hiệu chỉnh Benjamini Hochberg; so sánh Logistic Regression và Random Forest bằng ROC AUC, PR AUC, Precision, Recall và F1. Các phép kiểm tra độ nhạy dùng RobustScaler và mã hóa One Hot hai biến WorkLifeBalance, StockOptionLevel.

Kết quả chỉ là các mối liên hệ thống kê trong dữ liệu giả lập. Mô hình không được dùng để đưa ra quyết định bất lợi với từng người lao động.
