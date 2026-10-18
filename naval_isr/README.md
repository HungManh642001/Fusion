# Naval Multi-Source ISR Reconnaissance Data Fusion (naval_isr)

Dự án độc lập nghiên cứu, mô phỏng và phát triển hệ thống **Hợp nhất dữ liệu trinh sát tàu biển đa nguồn (Naval ISR Data Fusion)**.

---

## Cấu trúc Dự án

```
naval_isr/
├── docs/
│   ├── NAVAL_ISR_SYSTEM_ARCHITECTURE_GUIDE.md  # Tài liệu kỹ thuật chi tiết & Activity Diagrams
│   ├── TECHNICAL_SPEC.md                      # Đặc tả kỹ thuật kiến trúc hệ thống
│   └── IMPLEMENTATION_PLAN.md                 # Kế hoạch triển khai theo chuẩn TDD
├── src/
│   └── naval_isr/
│       ├── simulation/                        # Phase 1: Mô phỏng Radar, ESM, AIS, IMM filter, JPDA, CI
│       ├── sar_recon/                         # Phase 2: Viễn thám vệ tinh SAR, CA-CFAR, AIS Dead-Reckoning
│       ├── ai_inference/                      # Phase 3: PyTorch ST-GNN, DSmT PCR5, Threat Assessor
│       └── visualization/                     # Trực quan hóa bản đồ tác chiến 2D & Folium GIS
├── tests/                                     # Toàn bộ 18 bài kiểm thử tự động (Unit Tests)
├── scripts/
│   ├── run_phase1_simulation.py               # Chạy mô phỏng ven bờ Phase 1
│   ├── run_phase2_sar_demo.py                 # Chạy trinh sát vệ tinh SAR Phase 2
│   ├── export_phase3_colab.py                 # Xuất notebook Colab GPU Phase 3
│   └── evaluate_isr_metrics.py                # Đánh giá tổng hợp toàn diện các chỉ số
├── notebooks/
│   └── naval_isr_phase3_gpu_training.ipynb    # Notebook huấn luyện GPU độc lập
└── output/                                    # Kết quả đồ thị, bản đồ GIS HTML và báo cáo
```

---

## Hướng Dẫn Vận Hành Nhanh

### 1. Chạy toàn bộ Unit Tests (18 tests)
```bash
# Từ trong thư mục naval_isr:
pytest tests/ -v

# Hoặc từ thư mục gốc của repository:
pytest naval_isr/tests/ -v
```

### 2. Chạy kịch bản Giai đoạn 1 (Radar + ESM + AIS)
```bash
python naval_isr/scripts/run_phase1_simulation.py
```
*Đầu ra:* `naval_isr/output/phase1_tactical_display.png` và `phase1_ospa_metric.png`.

### 3. Chạy kịch bản Giai đoạn 2 (Ảnh vệ tinh SAR & Dark Vessels)
```bash
python naval_isr/scripts/run_phase2_sar_demo.py
```
*Đầu ra:* `naval_isr/output/tactical_sar_map.html` (mở bằng trình duyệt để xem bản đồ tương tác).

### 4. Đánh giá toàn diện các chỉ số định lượng
```bash
python naval_isr/scripts/evaluate_isr_metrics.py
```
*Đầu ra:* `naval_isr/output/naval_isr_master_report.md`.

### 5. Huấn luyện mô hình ST-GNN trên Google Colab GPU
- Mở [Google Colab](https://colab.research.google.com/) và tải lên file `naval_isr/notebooks/naval_isr_phase3_gpu_training.ipynb`.
- Bật GPU T4 và bấm Run All.
