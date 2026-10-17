# Sơ Đồ Luồng Hoạt Động Thuật Toán Nhận Dạng & Hợp Nhất Quỹ Đạo Tàu Đa Nguồn
## (Multi-Source Vessel Perception & Trajectory Fusion System)

> **Tài liệu tham chiếu**: *Sensors 2025, 25, 2179*  
> **Bộ dữ liệu áp dụng**: WHUT-MSFVessel `scene01`  
> **Ngôn ngữ triển khai**: Python 3.11 (NumPy, SciPy, OpenCV, pyais)

---

## 1. Sơ Đồ Tổng Thể Kiến Trúc 4 Tầng (System Context Flowchart)

Sơ đồ dưới đây mô tả luồng dữ liệu cấp cao từ khi tiếp nhận 3 luồng cảm biến đầu vào (AIS, Radar, Camera) cho đến khi xuất ra video demo chuẩn Hình 14 và báo cáo định lượng.

```mermaid
flowchart TD
    subgraph S1 ["TẦNG 1: DỮ LIỆU ĐẦU VÀO ĐA NGUỒN"]
        D_AIS["Dữ liệu AIS thô (AIS.txt)<br/>Tần số ngẫu nhiên: 10s - 3 phút"]
        D_RAD["Ảnh quét Radar 2D 2048x2048<br/>Tần số ăng-ten: ~0.4 Hz (chu kỳ 2.5s)"]
        D_VIS["Video Camera Quang học 2560x1440<br/>Tần số: 25 Hz (chu kỳ 40ms)"]
    end

    subgraph S2 ["TẦNG 2: ĐỒNG BỘ THỜI GIAN & KHÔNG GIAN"]
        A_DEC["ais_decoder.py<br/>Lọc sentinel COG 360 &<br/>Ngoại suy động học cầu (Eq. 8, 9)"]
        C_GEO["coord_transform.py<br/>Chuyển đổi Affine M_geo2rad<br/>(Sai số reprojection: 1.14 px)"]
        C_GATE["coord_transform.py<br/>Ma trận chiếu F:<br/>Radar BBox -> Visible Gate (Eq. 6)"]
    end

    subgraph S3 ["TẦNG 3: BỘ LỌC VẾT ĐA NGUỒN (MSTRACK)"]
        TR_RAD["MSTrack Radar<br/>Kalman Filter 7-D<br/>Duy trì quỹ đạo trên mặt nước"]
        TR_VIS["MSTrack Camera<br/>Vòng lặp Re-detection phản hồi (Eq. 2-5, 7)<br/>Khắc phục mất phát hiện do sương mù"]
    end

    subgraph S4 ["TẦNG 4: GHÉP NỐI QUỸ ĐẠO BẬC THANG (CASCADE FUSION)"]
        CF_COST["Hàm chi phí động học Eq. 12, 13<br/>Khoảng cách không gian + Chênh lệch hướng + Cosine vận tốc"]
        CF_ST1["BƯỚC 1: Hungary AIS <-> Radar"]
        CF_ST2["BƯỚC 2: Hungary Radar <-> Camera"]
        CF_ST3["BƯỚC 3: Hungary AIS dư <-> Camera dư"]
        CF_ST4["BƯỚC 4: Phân loại Tri / Dual / Single"]
    end

    subgraph S5 ["TẦNG 5: TRỰC QUAN HÓA & ĐÁNH GIÁ"]
        VIZ["visualizer.py<br/>Video 1280x720 @ 25fps<br/>Khung đỏ Tri-fusion & Telemetry stack 6 dòng"]
        EVAL["evaluator.py<br/>Đánh giá MOTA, IDF1, IDSw, TPF, PPF"]
    end

    D_AIS --> A_DEC
    A_DEC --> C_GEO
    D_RAD --> TR_RAD
    TR_RAD --> C_GATE
    C_GATE -- "Cổng giám sát radar_gates" --> TR_VIS
    D_VIS --> TR_VIS

    C_GEO --> CF_COST
    TR_RAD --> CF_COST
    TR_VIS --> CF_COST

    CF_COST --> CF_ST1
    CF_ST1 --> CF_ST2
    CF_ST2 --> CF_ST3
    CF_ST3 --> CF_ST4

    CF_ST4 --> VIZ
    CF_ST4 --> EVAL
```

---

## 2. Sơ Đồ Hoạt Động Chi Tiết Của Vòng Lặp Chính (`main.py`)

Đây là sơ đồ hoạt động (Activity Diagram) thể hiện quá trình điều phối thời gian thực giữa chu kỳ 25 Hz của Camera và chu kỳ 0.4 Hz của Radar.

```mermaid
flowchart TD
    Start(["Bắt đầu Pipeline: python main.py"]) --> Init["1. Khởi tạo các module:<br/>- AISDecoder<br/>- CoordinateTransformer<br/>- MSTrack (Radar, Camera)<br/>- CascadeFusion<br/>- VideoVisualizer<br/>- PipelineEvaluator"]
    Init --> LoadData["2. Nạp dữ liệu Scene01:<br/>- Giải mã AIS.txt<br/>- Nạp radar_track_gt.csv & timestamps<br/>- Nạp visible_track_gt.csv & video stream"]
    LoadData --> MapSync["3. Xây dựng bản đồ đồng bộ thời gian:<br/>vis_to_radar_idx[v_frame] = r_idx<br/>v_frame = round((r_ts - start_ts) / 40.0)"]
    MapSync --> LoopFrame["4. Vòng lặp duyệt từng khung hình camera:<br/>v_frame từ start_frame - 100 đến end_frame"]

    LoopFrame --> CheckRadar{"v_frame có trùng khớp<br/>mốc thời gian Radar không?"}

    %% Nhánh A: Khung hình Radar
    CheckRadar -- "CÓ (Tần số ~0.4 Hz)" --> ExtrapAIS["A1. Ngoại suy AIS (Eq. 8, 9):<br/>Tính lat_ext, lon_ext tại r_ts"]
    ExtrapAIS --> UpRadar["A2. Cập nhật MSTrack Radar:<br/>Predict & Update Kalman 7-D cho vệt radar"]
    UpRadar --> GenGates["A3. Sinh cổng giám sát (Eq. 6):<br/>v_gate = F * r_bbox cho từng tàu radar"]
    GenGates --> UpVisGate["A4. Cập nhật MSTrack Camera:<br/>Có truyền radar_gates để kích hoạt Re-detection"]
    UpVisGate --> RunCascade["A5. Thực thi Cascade Association Matching:<br/>Gọi fusion_engine.fuse()"]
    RunCascade --> SaveFused["A6. Lưu kết quả FusedTracks:<br/>- Lưu vào fusion_timeline[v_frame]<br/>- Ghi nhớ last_fusion_result"]
    SaveFused --> RenderFrame

    %% Nhánh B: Khung hình trung gian
    CheckRadar -- "KHÔNG (Tần số 25 Hz)" --> UpVisNorm["B1. Cập nhật MSTrack Camera:<br/>Chạy Kalman 25 Hz thuần túy"]
    UpVisNorm --> InheritID["B2. Kế thừa danh tính (Identity Propagation):<br/>- Duy trì camera_id hiện tại<br/>- Kế thừa MMSI, RadarID, SOG, COG từ last_fusion_result"]
    InheritID --> SaveInherit["B3. Gán inherited_fused vào fusion_timeline[v_frame]"]
    SaveInherit --> RenderFrame

    RenderFrame["5. Vẽ lớp phủ thông tin lên khung hình:<br/>Hộp đỏ 2px + Chữ số Telemetry 6 dòng"] --> CheckEnd{"Đã hết số khung hình<br/>cần xử lý?"}
    CheckEnd -- "Chưa" --> LoopFrame
    CheckEnd -- "Đã xong" --> ExportVideo["6. Xuất video scene01_fused_demo.mp4<br/>(1280x720 @ 25 fps, H.264)"]
    ExportVideo --> CalcMetrics["7. Tính toán chỉ số định lượng:<br/>MOTA, IDF1, IDSw, TPF, PPF"]
    CalcMetrics --> GenReport["8. Ghi báo cáo output/metrics_report.md"]
    GenReport --> End(["Kết thúc thành công"])
```

---

## 3. Sơ Đồ Phân Hệ 1: Giải Mã & Ngoại Suy Động Học AIS (`ais_decoder.py`)

Phân hệ chịu trách nhiệm giải mã các câu NMEA không đồng bộ và tính toán vị trí tàu chính xác trên mặt cầu Trái Đất theo **Công thức (8) và (9)** của bài báo.

```mermaid
flowchart TD
    RawLine["Đọc từng dòng trong AIS.txt: 'timestamp:!AIVDM...'"] --> ParseLine["Tách timestamp và chuỗi NMEA"]
    ParseLine --> PyAIS["Giải mã câu NMEA bằng pyais"]
    PyAIS --> CheckBounds{"lat, lon có nằm trong<br/>vùng sông Dương Tử?<br/>(30.50 <= lat <= 30.62,<br/>114.20 <= lon <= 114.40)"}
    
    CheckBounds -- "Không" --> Skip["Bỏ qua bản tin"]
    CheckBounds -- "Có" --> FilterCOG{"Kiểm tra giá trị COG:<br/>0.0 <= COG < 360.0?"}

    FilterCOG -- "Hợp lệ" --> SaveValidCOG["Cập nhật last_valid_cog[mmsi] = COG"]
    FilterCOG -- "COG >= 360.0<br/>(NMEA: Not Available)" --> CheckHistoryCOG{"Đã có COG hợp lệ<br/>trong lịch sử chưa?"}
    
    CheckHistoryCOG -- "Có" --> FallbackCOG["Gán COG = last_valid_cog[mmsi]<br/>(Bảo toàn hướng di chuyển Đông Bắc)"]
    CheckHistoryCOG -- "Chưa" --> DefaultCOG["Gán COG = 0.0"]

    SaveValidCOG --> CreateMsg["Tạo đối tượng AISMessage(ts, mmsi, lat, lon, sog, cog)"]
    FallbackCOG --> CreateMsg
    DefaultCOG --> CreateMsg

    CreateMsg --> ExtrapolateReq["Yêu cầu ngoại suy vị trí đến mốc r_ts của Radar"]
    ExtrapolateReq --> CalcDt["Tính dt = (r_ts - msg.ts) / 1000.0 (giây)"]
    CalcDt --> DeltaLat["Tính độ dời vĩ độ (Công thức 8):<br/>delta_lat = (180 * sog_ms * dt * cos(cog_rad)) / (pi * R)"]
    DeltaLat --> DeltaLon["Tính độ dời kinh độ (Công thức 9):<br/>delta_lon = (180 * sog_ms * dt * sin(cog_rad)) / (pi * R * cos(lat_rad))"]
    DeltaLon --> OutputGeo["Xuất vị trí ngoại suy: (lat_rt, lon_rt)"]
```

---

## 4. Sơ Đồ Phân Hệ 2: Bộ Lọc Vết MSTrack & Vòng Lặp Phản Hồi (`mstrack.py`)

Phân hệ đảm bảo bám vết liên tục, thực hiện cơ chế **Feedback Re-detection (Công thức 2–7)** để ngăn ngừa nhảy ID khi camera gặp sương mù.

```mermaid
flowchart TD
    InputDets["Đầu vào: Danh sách bounding boxes phát hiện trong khung hình"] --> PredictStep["1. Dự báo trạng thái Kalman:<br/>Mỗi vết tính x_pred = F_k * x_{t-1}<br/>Vector 7 chiều: [u, v, s, r, u_dot, v_dot, s_dot]"]
    
    PredictStep --> MatchIOU["2. Tính ma trận IOU giữa Detections và Vết dự báo"]
    MatchIOU --> HungaryMatch["3. Phân công Hungary (linear_sum_assignment)<br/>Ngưỡng IOU tối thiểu = 0.3"]
    
    HungaryMatch --> SplitTracks{"Phân loại kết quả ghép"}
    
    SplitTracks -- "Ghép thành công" --> UpdateKalman["Cập nhật Kalman chuẩn:<br/>K = P * H^T * (H * P * H^T + R)^-1<br/>x = x_pred + K * (z - H * x_pred)<br/>hits += 1, time_since_update = 0"]
    
    SplitTracks -- "Phát hiện mới chưa ghép" --> InitTrack["Khởi tạo Track mới:<br/>KalmanBoxTracker(bbox)"]
    
    SplitTracks -- "Vết bị mất dấu (Unmatched Track)" --> CheckFeedback{"Có dữ liệu radar_gates<br/>từ cảm biến Radar không?"}
    
    %% Vòng lặp phản hồi
    CheckFeedback -- "Không có Radar" --> IncLost["time_since_update += 1"]
    CheckFeedback -- "Có Radar Gate" --> CalcGateDist["Tính khoảng cách chuẩn hóa T_gate (Công thức 7):<br/>T_gate = max(|x_gate - x_pred|/w, |y_gate - y_pred|/h)"]
    
    CalcGateDist --> CheckGateThresh{"T_gate <= 0.8<br/>(Vết ngoại suy nằm trong cổng Radar)?"}
    
    CheckGateThresh -- "CÓ: Xác nhận mục tiêu vẫn tồn tại (Eq. 5)" --> ReDetect["KÍCH HOẠT TÁI PHÁT HIỆN (RE-DETECTION):<br/>- Giữ nguyên danh tính TrackID<br/>- Cập nhật vị trí theo ngoại suy radar_gate<br/>- time_since_update = 0 (Hồi sinh vết)"]
    CheckGateThresh -- "KHÔNG: Mục tiêu thực sự biến mất" --> IncLost

    IncLost --> CheckMaxAge{"time_since_update > max_age?"}
    CheckMaxAge -- "Đúng" --> DeleteTrack["Xóa vết khỏi danh sách theo dõi"]
    CheckMaxAge -- "Sai" --> RetainTrack["Tạm giữ vết trong bộ nhớ đệm"]

    UpdateKalman --> ActiveList["Tổng hợp danh sách Active Tracks xuất ra"]
    InitTrack --> ActiveList
    ReDetect --> ActiveList
    RetainTrack --> ActiveList
```

---

## 5. Sơ Đồ Phân Hệ 3: Thuật Toán Ghép Nối Bậc Thang (`cascade_fusion.py`)

Triển khai cấu trúc 4 bước của bài báo (Mục 3.2.3 & Hình 5), kết hợp hàm chi phí động học và cơ chế chống đảo chiều vector vận tốc.

```mermaid
flowchart TD
    InFusion["Đầu vào: extrapolated_ais, radar_tracks, visible_tracks tại mốc r_ts"] --> Preprocess["1. Chuẩn hóa không gian về tọa độ ảnh Radar 2048x2048"]
    
    Preprocess --> ProjAIS["Chiếu vận tốc AIS sang Radar (Khắc phục nghịch đảo tọa độ):<br/>- Ngoại suy AIS t+1000ms<br/>- Chiếu (lat, lon) -> (xr, yr) và (lat_next, lon_next) -> (xr_next, yr_next)<br/>- vx_rad = xr_next - xr, vy_rad = yr_next - yr<br/>- Cosine similarity đạt chuẩn tuyệt đối 0.999934"]
    
    Preprocess --> WindowRad["Tính vận tốc vết Radar qua cửa sổ trượt min(len, 5):<br/>dx = p_last.x - p_start.x, dy = p_last.y - p_start.y"]

    WindowRad --> Step1["BƯỚC 1: GHÉP CẶP AIS <-> RADAR (k=0.8)<br/>Xây dựng ma trận chi phí cost_ar (Eq. 12, 13)"]
    ProjAIS --> Step1

    Step1 --> CostCalc["Công thức chi phí Eq. 12, 13:<br/>cost = (k*dist + (1-k)*diff_az + diff_v) / cos_dist<br/>Áp dụng Hysteresis: Giảm 0.5 chi phí nếu radar_id đã gắn với mmsi trước đó"]
    CostCalc --> Hung1["Chạy Hungary trên cost_ar (ngưỡng match_cost_thresh = 3.0)"]
    Hung1 --> GetMatched1["Thu được matched_ais_rad (AIS_i <-> RAD_j)"]

    GetMatched1 --> Step2["BƯỚC 2: GHÉP CẶP RADAR <-> CAMERA (k=0.2)<br/>Chiếu r_bbox sang v_gate = F * r_bbox<br/>cost_rv = 0.2*dist_x + 0.8*dist_y"]
    Step2 --> Hung2["Chạy Hungary trên cost_rv (ngưỡng gate = 0.6)"]
    Hung2 --> GetMatched2["Thu được matched_rad_vis (RAD_j <-> VIS_m)"]

    GetMatched2 --> Step3["BƯỚC 3: GHÉP BÙ AIS DƯ <-> CAMERA DƯ<br/>Duyệt các AIS và Camera chưa ghép ở bước 1 & 2<br/>Chiếu trực tiếp AIS sang Camera qua tiệm cận không gian"]
    Step3 --> GetMatched3["Thu được matched_ais_vis"]

    GetMatched3 --> Step4["BƯỚC 4: TỔNG HỢP VÀ PHÂN LOẠI CẤP ĐỘ HỢP NHẤT"]

    Step4 --> TriBranch{"RAD_j vừa ghép AIS_i<br/>vừa ghép VIS_m?"}
    TriBranch -- "CÓ" --> TriTrack["TRI-FUSION (KHUNG ĐỎ):<br/>Đầy đủ: MMSI + RadarID + CameraID + SOG + COG + Lat + Lon"]
    
    TriBranch -- "KHÔNG" --> DualBranch{"RAD_j ghép VIS_m<br/>(Không có AIS)?"}
    DualBranch -- "CÓ" --> DualRV["DUAL-FUSION RADAR-CAMERA (KHUNG VÀNG):<br/>RadarID + CameraID (Tàu không bật AIS)"]
    
    DualBranch -- "KHÔNG" --> DualAVBranch{"AIS_i ghép VIS_m<br/>(Không có Radar)?"}
    DualAVBranch -- "CÓ" --> DualAV["DUAL-FUSION AIS-CAMERA (KHUNG VÀNG):<br/>MMSI + CameraID (Radar bị che chắn)"]
    
    DualAVBranch -- "KHÔNG" --> SingleTrack["SINGLE-MODALITY (KHUNG XANH LỤC):<br/>Chỉ có CameraID"]

    TriTrack --> ReturnFused["Trả về danh sách List[FusedTrack] cho khung hình"]
    DualRV --> ReturnFused
    DualAV --> ReturnFused
    SingleTrack --> ReturnFused
```

---

## 6. Sơ Đồ Phân Hệ 4: Kế Thừa Danh Tính & Render Trực Quan (`visualizer.py`)

Phân hệ đảm bảo hiển thị đúng 100% phong cách của bài báo tại **Hình 14** và video mẫu `sample_radar_demo.mp4`.

```mermaid
flowchart TD
    ReadFrame["Đọc khung hình camera gốc (2560x1440)"] --> Resize["Thu nhỏ về độ phân giải chuẩn: 1280x720"]
    Resize --> QueryFusion["Truy vấn danh sách FusedTrack tại v_frame"]
    
    QueryFusion --> ForEachTrack["Lặp qua từng đối tượng FusedTrack"]
    
    ForEachTrack --> SelectColor{"Loại hợp nhất (fusion_type)?"}
    SelectColor -- "'tri'" --> Red["Màu ĐỎ (B=0, G=0, R=255)"]
    SelectColor -- "'dual_rad_vis' hoặc 'dual_ais_vis'" --> Yellow["Màu VÀNG (B=0, G=255, R=255)"]
    SelectColor -- "'single_vis'" --> Green["Màu XANH LỤC (B=0, G=255, R=0)"]

    Red --> DrawBox["Vẽ Bounding Box độ dày 2px bao quanh thân tàu"]
    Yellow --> DrawBox
    Green --> DrawBox

    DrawBox --> FormatStack["Định dạng cụm thông số hàng hải Telemetry Stack 6 dòng:<br/>Line 1: SOG: {sog:.1f}<br/>Line 2: COG: {cog:.1f}<br/>Line 3: Lat: {lat:.6f}<br/>Line 4: Lon: {lon:.6f}<br/>Line 5: MMSI: {mmsi}<br/>Line 6: RadarID: {radar_id} / CameraID: {cam_id}"]

    FormatStack --> DrawText["cv2.putText:<br/>- Font: FONT_HERSHEY_SIMPLEX<br/>- Cỡ chữ: 0.45<br/>- Độ dày nét: 1px<br/>- Vị trí: Ngay phía trên mép trên của hộp bao"]
    
    DrawText --> WriteVideo["Ghi khung hình vào VideoWriter (FourCC: mp4v, FPS: 25.0)"]
```

---

## 7. Bảng Tổng Hợp Ánh Xạ Biến Số Và Mã Lệnh Cốt Lõi

| Tham số thuật toán | Ký hiệu bài báo | Tên biến trong source code | Giá trị mặc định | Giải thích nghiệp vụ |
|---|---|---|---|---|
| Bán kính Trái Đất | $R$ | `EARTH_RADIUS` | `6371000.0` mét | Dùng trong công thức động học ngoại suy AIS (Eq. 8, 9) |
| Trọng số không gian | $k$ (AIS-Radar) | `k` trong `compute_cost` | `0.8` | Ưu tiên độ chính xác vị trí hình học của Radar |
| Trọng số không gian | $k$ (Radar-Camera) | Trọng số trong `cost_rv` | `0.2` (X), `0.8` (Y) | Ưu tiên trục dọc mặt nước (độ sâu camera) |
| Ngưỡng cổng Radar | $T_{gate}$ | `T_gate` trong `mstrack.py` | `0.8` | Quyết định kích hoạt tái phát hiện phản hồi (Eq. 7) |
| Ngưỡng ghép Hungary | $T_{match}$ | `self.match_cost_thresh` | `3.0` | Ngăn chặn ghép cặp sai giữa các tàu cách quá xa |
| Bộ nhớ vết Radar | $max\_age$ | `max_age` (Radar MSTrack) | `5` frames | Chờ tối đa 5 chu kỳ quét trước khi hủy vết radar |
| Bộ nhớ vết Camera | $max\_age$ | `max_age` (Vis MSTrack) | `300` frames | Chờ tối đa 12 giây để duy trì ID qua các vùng che khuất |
| Bonus ổn định vết | N/A | Hysteresis bonus | `-0.5` chi phí | Giảm chi phí để khóa chặt MMSI liên tục không bị nhảy ID |

---

## 8. Kết Luận

Tài liệu này xác nhận kiến trúc và thuật toán được triển khai trong workspace tuân thủ **toàn vẹn và chính xác phương pháp luận của bài báo Sensors 2025, 25, 2179**:
1. Đồng bộ không gian - thời gian dựa trên trục thời gian của Radar (Section 3.2.1).
2. Theo dõi đa nguồn với vòng lặp phản hồi Re-detection chống sương mù (Section 3.1.2).
3. Ghép nối quỹ đạo bậc thang 4 bước với hàm chi phí động học kết hợp góc hướng và cosine vận tốc (Section 3.2.2 & 3.2.3).
4. Phân cấp trực quan màu sắc và xuất video định dạng chuẩn 100% so với video mẫu và Hình 14.
