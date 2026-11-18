# Deploy demo Streamlit

Demo: dự báo giá chip DRAM/NAND bằng Random Forest & XGBoost (dùng lại `demo_core.py`).

- **Repo**: https://github.com/phamminhkhoi312006-netizen/chip-price-forecast
- **Deploy một-bấm**: https://share.streamlit.io/deploy?repository=phamminhkhoi312006-netizen/chip-price-forecast&branch=main&mainModule=streamlit_app.py
  (cần đăng nhập Streamlit Cloud bằng GitHub lần đầu, sau đó bấm **Deploy**)

## 1. Chạy thử ở máy

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Mở http://localhost:8501.

## 2. Đưa code lên GitHub

Repo hiện tại đã được `git init` + commit sẵn. Chỉ cần tạo repo trống trên GitHub rồi:

```bash
git remote add origin https://github.com/<username>/<repo>.git
git branch -M main
git push -u origin main
```

Nếu chưa commit:

```bash
git add .
git commit -m "Demo du bao gia chip DRAM/NAND"
```

Lưu ý: repo phải **public** nếu dùng Streamlit Community Cloud miễn phí.
Các file cần có: `streamlit_app.py`, `demo_core.py`, `train_models.py`, `requirements.txt`,
`data/*.csv`, `results*/best_params.csv` (các file PNG/CSV khác không bắt buộc).

## 3. Deploy trên Streamlit Community Cloud

1. Vào https://share.streamlit.io → đăng nhập bằng GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Điền:
   - Repository: `<username>/<repo>`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - (Advanced settings) Python version: **3.12** (hoặc 3.13)
4. **Deploy**. Lần đầu cài dependency mất 2–5 phút.

Không cần secrets hay biến môi trường: app đọc dữ liệu trong `data/` và tham số trong `results*/`.

## 4. Sau khi deploy

- Link dạng `https://<app-name>.streamlit.io`.
- App cache kết quả huấn luyện theo `horizon`; lần đầu mở mỗi mốc dự báo mất ~10–20 giây,
  các lần sau gần như tức thì.
- Khi cập nhật `data/*.csv` hoặc `results*/best_params.csv`: commit + push, app tự chạy lại.
  Nếu số liệu không đổi, bấm **⋮ → Rerun** / **Clear cache** trong menu app.
- Nút **Chạy backtest** huấn luyện lại nhiều lần (6–36 tháng × 2 mô hình × 2 chip). Trên
  gói miễn phí (1 GB RAM) nên chọn ≤ 24 tháng để tránh chậm.

## 5. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `ModuleNotFoundError: xgboost` | Thiếu dòng trong `requirements.txt` hoặc chưa push file lên repo |
| `FileNotFoundError: market.csv` | Chưa push thư mục `data/` (kiểm tra `.gitignore`) |
| App bị "stuck" khi load | Lần đầu đang huấn luyện; xem log trong **Manage app**. Nếu quá 1–2 phút, kiểm tra phiên bản Python/pandas |
| Kết quả khác máy local | Chênh lệch phiên bản thư viện; xem log build để biết version đã cài |
| `tm.HORIZON`/cache lệch mốc | Bấm **Clear cache** rồi chọn lại tháng dự báo |

## 6. Ghi chú nội dung

Đã có tham số tinh chỉnh + `metrics.csv` + `forecast.csv` cho **tất cả các mốc h = 1…12**
(`results/` cho h=1, `results_h<h>/` cho h≥2).
Chọn mốc nào app cũng dùng đúng bộ tham số đã tune.
Muốn chạy lại một mốc: `HORIZON=<h> python train_models.py`.
Cache của app được khóa theo mtime các file trong `results*/`, nên push kết quả mới là app tự tính lại —
không cần bấm Clear cache.
