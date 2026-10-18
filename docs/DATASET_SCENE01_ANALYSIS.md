# Phân Tích Chi Tiết Bộ Dữ Liệu `datasets/scene01` (WHUT-MSFVessel)
## Mô Hình Hợp Nhất Đa Cảm Biến AIS - Marine Radar - Visible Camera Cho Giám Sát Tàu Thuyền
> **Tài liệu tham chiếu chính**: *Surface Vessels Detection and Tracking Method and Datasets with Multi-Source Data Fusion in Real-World Complex Scenarios*, Sensors 2025, 25(7), 2179.  
> **Tác giả nghiên cứu**: Wenbin Huang, Hui Feng, Haixiang Xu, Xu Liu, Jianhua He, Langxiong Gan, Xiaoqian Wang, Shanshan Wang (Đại học Công nghệ Vũ Hán - WHUT).  
> **Bộ dữ liệu**: WHUT-MSFVessel (Phân đoạn `scene01`).

---

## 1. Tổng Quan Về Bộ Dữ Liệu `datasets/scene01`

### 1.1 Bối Cảnh Và Nguồn Gốc Bộ Dữ Liệu WHUT-MSFVessel
Bộ dữ liệu **WHUT-MSFVessel** là bộ dữ liệu đo đạc thực tế đa nguồn đầu tiên trên thế giới phục vụ nghiên cứu nhận dạng, bám vết và hợp nhất quỹ đạo tàu thủy trong môi trường giao thông thủy phức tạp. Bộ dữ liệu được nhóm nghiên cứu thuộc *Phòng thí nghiệm Trọng điểm Công nghệ Tàu thủy Hiệu năng cao (Đại học Công nghệ Vũ Hán - WHUT)* thu thập liên tục trong 12 tháng tại khu vực sông Dương Tử (Yangtze River, Vũ Hán, Trung Quốc).

Trạm thu thập dữ liệu đặt cố định trên bờ, vuông góc với luồng hàng hải chính:
- **Tọa độ địa lý trạm (WGS-84)**: `[114.306087° E, 30.563549° N]`.
- **Độ cao triển khai ($h$)**: `20 mét` so với mặt nước sông.
- **Trường quan sát (Field of View - FOV)**: Bao quát toàn bộ luồng sông rộng khoảng 1.5 - 2 km, với tầm quét radar đạt 5 km và vùng phủ sóng AIS lên đến 10 km.

```
                              [Trạm Quan Sát Bờ (20m)]
                                 [114.306087, 30.563549]
                                      /     |     \
                                     /      |      \
              Visible Camera (54.3° FOV)    |       Marine Radar (360° PPI, 5km)
                                   /        |        \
                                  /         |         \
============================= Luồng Sông Dương Tử (Yangtze River) =============================
    [Tàu A: Có AIS + Radar + Camera]      [Tàu B: Không bật AIS (Radar + Camera)]    [Tàu nhỏ C (Chỉ Camera)]
================================================================================================
                                      \     |     /
                                     AIS Receiver (10 km, NAIS-500)
```

### 1.2 Đặc Thù Riêng Của Phân Đoạn `datasets/scene01`
Trong tập dữ liệu gồm 22 phân đoạn (`scene01` đến `scene22`), `scene01` là phân đoạn kiểm thử chuẩn (Benchmark Test Scene) đại diện cho kịch bản thời tiết nắng ráo, tầm nhìn quang học tốt nhưng mật độ tàu di chuyển đan xen cao trên luồng sông hẹp:
- **Thời lượng ghi hình**: 09 phút 06 giây (546 giây = 13,650 khung hình camera @ 25 fps; 228 khung quét radar).
- **Điều kiện thời tiết**: Nắng (Sunny), độ chiếu sáng cao, mặt nước gợn sóng phản xạ nhẹ.
- **Số lượng mục tiêu Radar ($N_R$)**: 12 tàu xuất hiện trong phạm vi quét 5 km của radar.
- **Số lượng mục tiêu Camera ($N_V$)**: 4 tàu di chuyển qua trường nhìn quang học $54.3^\circ$ của camera (Track IDs: 0, 1, 2, 3).
- **Số mục tiêu đồng thời tối đa trong góc nhìn camera ($MN$)**: 3 tàu.
- **Số mục tiêu bị che khuất ($OC$)**: 1 tàu (tàu di chuyển phía sau tàu lớn khác, gây mất dấu tạm thời trên camera).

### 1.3 Cấu Trúc Thư Mục Của `datasets/scene01`
```
datasets/scene01/
├── AIS.txt                 # Tệp văn bản chứa chuỗi bản tin NMEA AIS thô kèm timestamp mili-giây
├── radar_images/           # 228 tệp ảnh vòng quét PPI Radar 2D (kích thước 2048 x 2048 pixel)
│   ├── 2364714282.jpg      # Đặt tên tệp theo timestamp mili-giây (chu kỳ quét ~2.24s)
│   ├── 2364716530.jpg
│   └── ...
├── radar_track_gt.csv      # Nhãn Ground Truth bám vết mục tiêu Radar theo chuẩn MOT Challenge (2,080 dòng)
├── radar_video.mp4         # Video dựng sẵn ghép từ chuỗi ảnh quét Radar (tần số ~0.45 Hz)
├── visible_images/         # 899 ảnh trích xuất từ camera quang học (độ phân giải 2560 x 1440)
│   ├── 2364716724_123.20_1.20_1.00.jpg  # Tên tệp: <Timestamp>_<Pan>_<Tilt>_<Zoom>.jpg
│   └── ...
├── visible_track_gt.csv    # Nhãn Ground Truth bám vết Camera theo chuẩn MOT Challenge (18,296 dòng)
└── visible_video.mp4       # Video camera độ phân giải cao 2560 x 1440 @ 25 fps (13,650 frames)
```

---

## 2. Phân Tích Chi Tiết 3 Loại Dữ Liệu: Thông Tin Được Trích Xuất & Cách Bài Báo Sử Dụng

### 2.1 Loại Dữ Liệu 1: Hệ Thống Nhận Dạng Tự Động AIS (`AIS.txt`)

#### A. Thiết Bị Thu Thập & Đặc Tính Kỹ Thuật
- **Thiết bị**: Máy thu phát Simrad NAIS-500 Class B AIS.
- **Tần số hoạt động**: Hai kênh VHF hàng hải chuẩn quốc tế 161.975 MHz (Kênh 87B) và 162.025 MHz (Kênh 88B).
- **Phạm vi nhận tín hiệu**: Bán kính xấp xỉ 10 km xung quanh trạm quan sát.
- **Chu kỳ gửi tin**: Không đồng đều, phụ thuộc vào vận tốc tàu và loại trạm phát Class A/Class B (dao động từ 10 giây đến vài phút một bản tin).

#### B. Định Dạng Tệp Dữ Liệu Gốc
Tệp `AIS.txt` lưu trữ các bản tin dạng văn bản thô theo chuẩn **NMEA 0183** (chủ yếu là câu `!AIVDM` - bản tin từ tàu khác, và `!AIVDO` - bản tin trạm sở tại), được tiền tố hóa bằng dấu thời gian tính bằng mili-giây:
```text
<Timestamp_ms>:<NMEA_Sentence>\r\n
```
*Ví dụ thực tế trong `datasets/scene01/AIS.txt`:*
```text
2364714637:!AIVDM,1,1,,A,B6:bDwP0?22lBbTH3AHBcwrW;P06,0*12
2364714872:!AIVDM,1,1,,A,16:c1h5P008;;M0AOT?P00Id1P00,0*78
2364715076:!AIVDM,2,1,0,B,56:bFT@000030000000hV1PTr0QDv3KGSP000016883390@P035mB0CP0000,0*2
2364715086:!AIVDM,2,2,0,B,00000000000,2*27
```

#### C. Các Trường Thông Tin Được Giải Mã & Trích Xuất
Sau khi giải mã trường mã hóa 6-bit ASCII NMEA (Packet Type 1, 2, 3 cho vị trí tàu loại Class A, và Type 18/19 cho Class B, Type 5 cho thông tin tĩnh), bài báo trích xuất và sử dụng 5 trường dữ liệu hạt nhân:

| Tên trường | Kiểu dữ liệu | Đơn vị | Ý nghĩa chi tiết | Vai trò trong thuật toán của bài báo |
|---|---|---|---|---|
| `ts` (`timestamp`) | `int64` | ms | Dấu thời gian Unix mili-giây khi trạm nhận bản tin | Trục mốc thời gian để tính độ trễ $\Delta t$ phục vụ ngoại suy động học |
| `mmsi` | `int32` | 9 chữ số | Mã nhận dạng dịch vụ di động hàng hải duy nhất toàn cầu (Maritime Mobile Service Identity) | Định danh toàn cầu định đoạt danh tính tàu, liên kết với Radar ID và Camera ID |
| `lat` (`latitude`) | `float64` | Độ thập phân | Vĩ độ địa lý WGS-84 (tại sông Dương Tử: ~30.50° đến 30.62° N) | Vị trí không gian để chiếu sang ảnh radar/camera |
| `lon` (`longitude`) | `float64` | Độ thập phân | Kinh độ địa lý WGS-84 (tại sông Dương Tử: ~114.20° đến 114.40° E) | Vị trí không gian để chiếu sang ảnh radar/camera |
| `sog` | `float32` | knots (hải lý/h) | Tốc độ trên mặt đất của tàu (Speed Over Ground). $1\text{ knot} = 0.514444\text{ m/s}$ | Dùng trong công thức động học ngoại suy vị trí và tính hàm chi phí động học |
| `cog` | `float32` | Độ ($0^\circ - 360^\circ$) | Hướng di chuyển trên mặt đất của tàu (Course Over Ground) | Góc định hướng vector vận tốc; xử lý giá trị $360.0$ (không khả dụng) bằng lịch sử |

#### D. Cách Bài Báo Xử Lý Và Ứng Dụng AIS Trong Hệ Thống
1. **Ngoại suy động học trên mặt cầu Trái Đất (Dead-Reckoning Extrapolation - Công thức 8 & 9)**:
   Do AIS có tần số thấp (10s - 3 phút), không thể dùng trực tiếp để so khớp với Radar (chu kỳ 2.24s). Bài báo ngoại suy vị trí thời gian thực $(lat_{rt}, lon_{rt})$ tại thời điểm quét $T_{rad}$ của Radar:
   $$\Delta t = \frac{T_{rad} - t_{last}}{1000} \quad (\text{giây})$$
   $$lat_{rt} = lat_{last} + \frac{180 \cdot (SOG \cdot 0.514444) \cdot \Delta t \cdot \cos(COG_{rad})}{\pi \cdot R}$$
   $$lon_{rt} = lon_{last} + \frac{180 \cdot (SOG \cdot 0.514444) \cdot \Delta t \cdot \sin(COG_{rad})}{\pi \cdot R \cdot \cos\left(\frac{\pi \cdot lat_{last}}{180}\right)}$$
   *Trong đó*: $R = 6,371,000\text{ m}$ là bán kính Trái Đất.
2. **Chiếu tọa độ địa lý sang tọa độ pixel Radar (Công thức 10)**:
   Dùng ma trận biến đổi Affine $M_{geo2rad}$ để ánh xạ $(lon, lat) \to (x_r, y_r)$ trên ảnh radar kích thước $2048 \times 2048$ với sai số chỉ $1.14\text{ pixel}$.
3. **Làm điểm hạt giống (Seed Point) hồi sinh vết Radar bị mất (Công thức 3)**:
   Nếu bám vết radar bị mất dấu do tín hiệu phản xạ suy giảm, tọa độ AIS ngoại suy nằm trong cổng bám vết $Bbox_{gate}^{rad}$ sẽ được dùng làm điểm gieo mầm $S$ để trích xuất vùng pixel liên thông $P(x,y) > T_p$, khôi phục lại hộp bao mục tiêu bị thiếu.
4. **Cung cấp dữ liệu vận tốc, hướng đi và ID quốc tế**:
   AIS bổ trợ thông tin ngữ nghĩa (Semantic & Kinematic) mà Radar và Camera đơn thuần không thể có (MMSI, SOG, COG).

---

### 2.2 Loại Dữ Liệu 2: Radar Hàng Hải (`radar_images/`, `radar_track_gt.csv`, `radar_video.mp4`)

#### A. Thiết Bị Thu Thập & Đặc Tính Kỹ Thuật
- **Thiết bị**: Simrad HALO20+ Pulse Compression Marine Radar (Radar nén xung hàng hải).
- **Phạm vi quan sát**: Tầm phát hiện tối đa 36 hải lý; trong thực nghiệm thiết lập bán kính quan sát hiệu dụng là 5 km.
- **Tần số quay ăng-ten**: Vận tốc quay cố định tạo chu kỳ quét xấp xỉ 2.24 - 2.5 giây/vòng (~0.45 Hz).
- **Định dạng dữ liệu**: Ảnh 2 chiều PPI (Plan Position Indicator) kích thước $2048 \times 2048$ pixel, 8-bit. Mỗi ảnh đại diện cho một mặt cắt quét $360^\circ$ luồng sông tại thời điểm tương ứng.

#### B. Cấu Trúc Tệp Nhãn Bám Vết Radar Ground Truth (`radar_track_gt.csv`)
Tệp nhãn ground truth được các chuyên gia hàng hải gán nhãn thủ công thông qua công cụ DarkLabel, tuân thủ định dạng chuẩn của bài toán bám vết đa mục tiêu **MOT Challenge**:
```text
<frame_id>, <track_id>, <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <conf>, <x_world>, <y_world>, <z_world>
```
*Ví dụ thực tế trong `datasets/scene01/radar_track_gt.csv`:*
```text
0, 3,  1214, 981, 25, 30, -1, -1, -1, -1
0, 1,   946, 850, 31, 27, -1, -1, -1, -1
0, 4,  1058, 864, 33, 31, -1, -1, -1, -1
0, 15, 1606, 1085, 34, 59, -1, -1, -1, -1
```

#### C. Ý Nghĩa Chi Tiết Từng Trường Trong Dữ Liệu Radar

| Trường dữ liệu | Kiểu | Ý nghĩa trong ngữ cảnh Radar hàng hải |
|---|---|---|
| `frame_id` | `int` | Chỉ số vòng quét radar (từ 0 đến 225, tổng cộng 226 vòng quét có nhãn). Mỗi frame tương ứng với một tệp ảnh trong `radar_images/` có timestamp mili-giây |
| `track_id` | `int` | Mã số định danh duy nhất của tàu trên màn quét radar xuyên suốt video (`scene01` có 12 tàu: ID 1, 3, 4, 5, 6, 8, 9, 10, 13, 15, 18, 26) |
| `bbox_left` | `float` | Tọa độ $x_{min}$ (pixel) góc trên-trái của hộp bao vết phản xạ radar trên ảnh $2048 \times 2048$ |
| `bbox_top` | `float` | Tọa độ $y_{min}$ (pixel) góc trên-trái của hộp bao vết phản xạ radar trên ảnh $2048 \times 2048$ |
| `bbox_width` | `float` | Chiều rộng hộp bao vết phản xạ radar (tương ứng kích thước bề ngang vệt vang sóng) |
| `bbox_height` | `float` | Chiều cao hộp bao vết phản xạ radar (tương ứng chiều dài thân tàu dọc theo tia quét) |
| `conf` | `float` | Độ tin cậy gán nhãn: Trong file GT là `-1` (hoặc `1`); trong mô hình phát hiện YOLOv8 là xác suất dự đoán $0.0 - 1.0$ |
| `x_world, y_world, z_world` | `float` | Tọa độ 3D thực tế; trong các bộ dữ liệu 2D radar trên mặt phẳng sông, các trường này được gán mặc định `-1` (bỏ qua) |

#### D. Cách Bài Báo Xử Lý Và Ứng Dụng Radar Trong Hệ Thống
1. **Làm Trục Quy Chiếu Thời Gian Chủ (Master Clock Synchronization)**:
   Chu kỳ quét của radar ($T_{rad} \approx 2.24\text{s}$) được chọn làm nhịp đập đồng bộ cho toàn bộ hệ thống. Mọi bản tin AIS và khung hình camera đều được căn chỉnh theo $T_{rad}$.
2. **Khắc phục nhược điểm "mù chiều sâu" của Camera**:
   Ảnh radar biểu diễn góc nhìn từ trên cao (Bird's Eye View - BEV). Khoảng cách từ tâm radar đến mục tiêu chính là khoảng cách thực tế trên mặt nước, không bị biến dạng phối cảnh như ảnh camera.
3. **Hoạt động xuyên thời tiết bất lợi**:
   Radar không bị cản trở bởi sương mù dày đặc, trời mưa hay ban đêm.
4. **Chiếu sinh cổng giám sát Camera ($Bbox_{gate}^{vis} = F \cdot Bbox_{det}^{rad}$ - Công thức 6)**:
   Vị trí và kích thước vệt radar được nhân với ma trận chiếu $F$ để tạo ra một "chiếc cổng" trên khung hình camera. Khi camera bị mất dấu do mưa hay sương mù, cổng radar này sẽ "chỉ đường" cho camera khôi phục vết tàu mà không bị nhảy ID.

---

### 2.3 Loại Dữ Liệu 3: Camera Quang Học (`visible_images/`, `visible_track_gt.csv`, `visible_video.mp4`)

#### A. Thiết Bị Thu Thập & Đặc Tính Kỹ Thuật
- **Thiết bị**: Camera PTZ Hikvision DS-2DE7440IW-A Network Speed Dome.
- **Cảm biến hình ảnh**: 1/2.8" Progressive Scan CMOS, zoom quang học 40x.
- **Trường nhìn ngang (Horizontal FOV)**: $54.3^\circ$, hướng ngang qua luồng sông Dương Tử.
- **Độ phân giải**: $2560 \times 1440$ pixel (chuẩn 2K QHD).
- **Tốc độ khung hình**: 25 Hz (chu kỳ 40 mili-giây/frame).

#### B. Định Dạng Tên Ảnh Khung Hình Rời (`visible_images/`)
Tên các tệp ảnh trong thư mục `visible_images/` có cấu trúc:
```text
<Timestamp_ms>_<Pan>_<Tilt>_<Zoom>.jpg
```
*Ví dụ*: `2364716724_123.20_1.20_1.00.jpg`
- `2364716724`: Dấu thời gian Unix mili-giây (đồng bộ với hệ thống thu thập).
- `123.20`: Góc quay ngang Pan của camera ($123.2^\circ$).
- `1.20`: Góc ngẩng dọc Tilt ($1.2^\circ$).
- `1.00`: Mức thu phóng quang học Zoom ($1.0\times$ - góc rộng cực đại).

#### C. Cấu Trúc Tệp Nhãn Bám Vết Camera Ground Truth (`visible_track_gt.csv`)
Tệp tuân thủ định dạng MOT Challenge:
```text
<frame_id>, <track_id>, <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <conf>, <x_world>, <y_world>, <z_world>
```
*Ví dụ thực tế trong `datasets/scene01/visible_track_gt.csv`:*
```text
0, 0,  431, 771, 181, 28, -1, -1, -1, -1
0, 1, 1391, 759, 163, 27, -1, -1, -1, -1
0, 2, 2168, 759, 113, 30, -1, -1, -1, -1
```

#### D. Ý Nghĩa Chi Tiết Từng Trường Trong Dữ Liệu Camera

| Trường dữ liệu | Kiểu | Ý nghĩa trong ngữ cảnh Camera giám sát |
|---|---|---|
| `frame_id` | `int` | Chỉ số khung hình video (từ 0 đến 13,649, tương ứng 546 giây video 25 fps) |
| `track_id` | `int` | Mã số định danh duy nhất của tàu xuất hiện trong video (`scene01` có 4 tàu: ID 0, 1, 2, 3) |
| `bbox_left` | `float` | Tọa độ $x_{min}$ (pixel) góc trên-trái hộp bao thân tàu trên khung hình $2560 \times 1440$ |
| `bbox_top` | `float` | Tọa độ $y_{min}$ (pixel) góc trên-trái hộp bao thân tàu trên khung hình $2560 \times 1440$ |
| `bbox_width` | `float` | Chiều rộng bounding box (chiều dài thân tàu nhìn từ góc ngang) |
| `bbox_height` | `float` | Chiều cao bounding box (chiều cao mạn tàu và cabin tàu) |
| `conf` | `float` | Độ tin cậy phát hiện: Trong GT là `-1`; trong mô hình phát hiện YOLOv8 là xác suất tin cậy $conf \in [0.0, 1.0]$ |
| `x_world, y_world, z_world` | `float` | Tọa độ 3D thực tế; trong tập dữ liệu gán mặc định `-1` |

#### E. Cách Bài Báo Xử Lý Và Ứng Dụng Camera Trong Hệ Thống
1. **Cung cấp góc phương vị quang học cực kỳ chuẩn xác**:
   Vị trí hoành độ $x$ trên ảnh phản ánh chính xác phương vị góc từ trạm quan sát đến tàu.
2. **Trích xuất đặc trưng ngoại quan trực quan (Re-ID)**:
   Mô hình DeepSORT trích xuất vector đặc trưng diện mạo (màu sắc, cấu trúc ca-bin, đường nét thân tàu) để phân biệt các tàu di chuyển gần nhau.
3. **Cơ chế tái phát hiện phản hồi (Feedback-Based Visible Tracking Refine - Công thức 4, 5, 7)**:
   Khi tàu đi vào vùng sương mù, mưa hoặc bị che khuất một phần, điểm tin cậy trung bình $mconf = \frac{1}{n}\sum conf_i$ bị sụt giảm khiến thuật toán thông thường bị mất vết. Khi đó, hệ thống lấy cổng giám sát từ Radar chiếu sang ($Bbox_{gate}^{vis}$), kiểm tra điều kiện chuẩn hóa:
   $$T_{gate} = \max\left(\frac{|x_{gate} - x_{pred}|}{w}, \frac{|y_{gate} - y_{pred}|}{h}\right) \le 0.8$$
   Nếu thỏa mãn, hệ thống giữ nguyên Track ID, khôi phục lại vết tàu và duy trì liên tục quỹ đạo mà không bị gãy ID (ngăn chặn ID Switch).
4. **Không gian hiển thị chính (Main Viewport)**:
   Toàn bộ kết quả hợp nhất (hộp màu phân loại và bảng thông số hải đồ Telemetry) được vẽ trực tiếp đè lên video camera, hỗ trợ trực quan hóa cho thuyền viên hoặc nhân viên VTS (Vessel Traffic Services).

---

## 3. Đầu Ra Của Hệ Thống (System Outputs) & Ý Nghĩa Chi Tiết Từng Trường Dữ Liệu

Quá trình hợp nhất đa nguồn sản sinh ra 4 nhóm kết quả đầu ra chính:
1. Đối tượng cấu trúc dữ liệu quỹ đạo hợp nhất (`FusedTrack`).
2. Tệp video trực quan hóa hợp nhất (`scene01_fused_demo.mp4`).
3. Bảng phân loại cấp độ hợp nhất (Multi-Source Association Classes).
4. Báo cáo các chỉ số định lượng đánh giá hiệu năng (Quantitative Evaluation Metrics Report).

```
[3 Luồng Dữ Liệu: AIS + Radar + Camera]
                   │
                   ▼
       [Pipeline Hợp Nhất Đa Nguồn]
                   │
  ┌────────────────┼────────────────┬────────────────┐
  ▼                ▼                ▼                ▼
[1. FusedTrack] [2. Video Demo]  [3. Cấp Độ]    [4. Báo Cáo Đo Đạc]
(Cấu trúc dữ     (Bounding Box   (Tri / Dual /  (MOTA, IDF1, IDSw,
 liệu bộ nhớ)     & Telemetry)    Single)        TPF, PPF, Accuracy)
```

---

### 3.1 Cấu Trúc Dữ Liệu Quỹ Đạo Hợp Nhất (`FusedTrack`)
Mỗi tàu sau khi được thuật toán Cascade Association kết nối thành công sẽ được biểu diễn bởi một đối tượng `FusedTrack` với các trường thông tin sau:

| Tên trường | Kiểu dữ liệu | Đơn vị / Định dạng | Ý nghĩa nghiệp vụ chi tiết |
|---|---|---|---|
| `fusion_id` | `int` | Số nguyên tự tăng | Mã định danh duy nhất của thực thể tàu hợp nhất trong toàn hệ thống |
| `fusion_type` | `string` | `'tri'`, `'dual_rad_vis'`, `'dual_ais_vis'`, `'single_vis'` | Loại hình hợp nhất (cấp độ tin cậy của quỹ đạo) |
| `mmsi` | `int` / `None` | 9 chữ số (ví dụ: `413806072`) | Mã nhận dạng hàng hải quốc tế lấy từ AIS. Nếu tàu không có AIS thì giá trị là `None` |
| `radar_id` | `int` / `None` | Số nguyên (ví dụ: `23` hoặc `26`) | Mã số vết bám của tàu trên màn quét radar (MSTrack Radar ID) |
| `camera_id` | `int` / `None` | Số nguyên (ví dụ: `20` hoặc `3`) | Mã số vết bám của tàu trên camera quang học (MSTrack Camera ID) |
| `camera_conf` | `float` | $0.0 - 1.0$ (ví dụ: `1.0` hoặc `0.85`) | Điểm số tin cậy phát hiện của camera tại khung hình hiện tại |
| `bbox_vis` | `List[float]` | `[x, y, w, h]` (pixel) | Tọa độ hộp bao tàu trên camera độ phân giải $2560 \times 1440$ |
| `bbox_rad` | `List[float]` | `[x, y, w, h]` (pixel) | Tọa độ hộp bao tàu trên màn quét radar $2048 \times 2048$ |
| `lat` | `float` | Độ thập phân (ví dụ: `30.570142`) | Vĩ độ thực tế WGS-84 của tàu tại thời điểm hiện tại |
| `lon` | `float` | Độ thập phân (ví dụ: `114.297007`) | Kinh độ thực tế WGS-84 của tàu tại thời điểm hiện tại |
| `sog` | `float` | Knots (ví dụ: `5.7`) | Tốc độ trên mặt đất của tàu (Speed Over Ground) |
| `cog` | `float` | Độ (ví dụ: `34.5°`) | Hướng di chuyển thực tế trên mặt đất của tàu (Course Over Ground) |

---

### 3.2 Đầu Ra Trực Quan Hóa (Video Demo & Telemetry Overlay)
Trong video xuất ra (`output/scene01_fused_demo.mp4`), hệ thống vẽ lớp thông tin đồ họa chuẩn hóa theo **Hình 14 của bài báo**:

#### A. Quy Tắc Mã Hóa Màu Sắc Bounding Box
- **Hộp màu ĐỎ (Red - `COLOR_TRI`)**: **Tri-fusion**. Mục tiêu được hợp nhất thành công trọn vẹn từ cả 3 nguồn (AIS + Radar + Camera). Đây là trạng thái tin cậy cao nhất, sở hữu đầy đủ danh tính quốc tế, vị trí địa lý, vận tốc và hình ảnh.
- **Hộp màu VÀNG (Yellow - `COLOR_DUAL`)**: **Dual-fusion**. Mục tiêu hợp nhất được từ 2 nguồn (Radar + Camera hoặc AIS + Camera). Thường xảy ra khi tàu nhỏ không trang bị bộ phát AIS hoặc khi radar gặp góc khuất bờ kè.
- **Hộp màu XANH LỤC (Green - `COLOR_SINGLE`)**: **Single-modality**. Mục tiêu chỉ phát hiện được duy nhất trên Camera (không có tín hiệu Radar và không có AIS).

#### B. Khối Thông Số Hải Đồ Đa Dòng (Telemetry Stack)
Phía trên mép trên của mỗi hộp bao màu đỏ, hệ thống hiển thị khối chữ gồm 6-7 dòng thông số định dạng như sau:
```text
SOG: 5.7
COG: 34.5
Lat: 30.570142
Lon: 114.297007
MMSI: 413806072
RadarID: 23
CameraID: 20-1.0
```
- **Dòng 1 (`SOG: 5.7`)**: Vận tốc tức thời của tàu (5.7 hải lý/giờ).
- **Dòng 2 (`COG: 34.5`)**: Hướng di chuyển hành trình góc $34.5^\circ$ (hướng Đông Bắc theo dòng sông).
- **Dòng 3 (`Lat: 30.570142`)**: Vĩ độ WGS-84 thời gian thực của tàu.
- **Dòng 4 (`Lon: 114.297007`)**: Kinh độ WGS-84 thời gian thực của tàu.
- **Dòng 5 (`MMSI: 413806072`)**: Mã nhận diện định danh AIS quốc tế của tàu.
- **Dòng 6 (`RadarID: 23`)**: Mã bám vết trên trạm radar.
- **Dòng 7 (`CameraID: 20-1.0`)**: Mã bám vết camera kèm độ tin cậy phát hiện (`1.0`).

---

### 3.3 Đầu Ra Báo Cáo Đo Đạc & Các Chỉ Số Định Lượng (Quantitative Evaluation Metrics)

Bài báo và mã nguồn phân chia hệ thống đánh giá thành 2 nhóm chỉ số:

#### Nhóm 1: Chỉ Số Đánh Giá Bám Vết Đa Mục Tiêu (Theo Chuẩn MOT Challenge - Bảng 3 & 4)
Đánh giá năng lực của bộ theo dõi `MSTrack` trên từng cảm biến đơn lẻ (Radar và Camera):

1. **MOTA (Multiple Object Tracking Accuracy - Công thức 14)**:
   $$\text{MOTA} = 1 - \frac{\sum (FN_t + FP_t + IDSw_t)}{\sum GT_t}$$
   - $FN_t$ (False Negatives): Số lượng mục tiêu bị bỏ sót phát hiện ở frame $t$.
   - $FP_t$ (False Positives): Số lượng phát hiện sai/nhiễu không có thật ở frame $t$.
   - $IDSw_t$ (Identity Switches): Số lần nhảy/đổi ID sai lệch trên cùng một con tàu.
   - $GT_t$ (Ground Truth): Tổng số lượng mục tiêu thực tế ở frame $t$.
   - *Ý nghĩa*: Thước đo toàn diện nhất về độ chính xác bám vết. Điểm càng gần 1.0 (100%) thuật toán càng hoàn hảo. Trong `scene01`, bài báo đạt MOTA = **0.941** trên camera và **0.875** trên radar.
2. **IDF1 (Identification F1-Score - Công thức 17)**:
   $$\text{IDF1} = \frac{2 \cdot IDP \cdot IDR}{IDP + IDR} = \frac{2 \cdot IDTP}{2 \cdot IDTP + IDFP + IDFN}$$
   - $IDTP$: Số lần mục tiêu được phát hiện đúng với đúng ID xuyên suốt.
   - $IDFP$: Phát hiện đúng vị trí nhưng bị gán sai ID.
   - $IDFN$: Mục tiêu thực tế bị bỏ sót không có ID.
   - *Ý nghĩa*: Đánh giá khả năng duy trì tính nhất quán của ID mục tiêu qua thời gian dài (không bị đổi số ID). Trong `scene01`, bài báo đạt IDF1 = **0.942**.
3. **IDSw (Identity Switches)**:
   - Tổng số lần tàu bị đổi mã số theo dõi do che khuất, gián đoạn. Càng nhỏ càng tốt (trong `scene01` chỉ có 3 lần nhảy ID).
4. **MT (Mostly Tracked Targets)**:
   - Tỷ lệ phần trăm các quỹ đạo tàu thực tế được bám vết thành công ít nhất $80\%$ vòng đời.
5. **ML (Mostly Lost Targets)**:
   - Tỷ lệ phần trăm các quỹ đạo tàu bị mất dấu quá $80\%$ vòng đời (được theo dõi dưới $20\%$).

#### Nhóm 2: Chỉ Số Đánh Giá Hợp Nhất Quỹ Đạo Đa Nguồn (Do Bài Báo Đề Xuất - Bảng 5)
Đánh giá mức độ chính xác khi ghép cặp dữ liệu giữa 3 cảm biến dị thể:

1. **TPF (True Positive Fusion Rate)**:
   - Tỷ lệ các trường hợp cả 3 nguồn (AIS, Radar, Camera) được kết nối và hợp nhất chính xác tuyệt đối với nhau (hộp màu đỏ). Trong `scene01` đạt 263 mẫu.
2. **PPF (Part Positive Fusion Rate)**:
   - Tỷ lệ các trường hợp kết nối đúng được 2 nguồn cảm biến nhưng nguồn thứ 3 bị thiếu hoặc kết nối sai lệch một phần (hộp màu vàng). Trong `scene01` đạt 0 mẫu (không có sai lệch bộ phận).
3. **FPF (False Positive Fusion Rate)**:
   - Tỷ lệ gán ghép sai giữa MMSI, Radar ID và Camera ID (ghép nhầm thông tin của tàu này sang tàu khác). Trong `scene01` chỉ có 4 trường hợp (rất thấp).
4. **FNF (False Negative Fusion Rate)**:
   - Tỷ lệ mục tiêu tồn tại nhưng hệ thống không thể hợp nhất (thường xảy ra ở thời điểm đầu khi tàu mới đi vào mép khung nhìn camera và cần thời gian khởi tạo). Trong `scene01` có 12 mẫu.
5. **Accuracy (Độ chính xác hợp nhất tổng hợp - Công thức 18)**:
   $$\text{Accuracy} = \frac{TPF + 0.5 \cdot PPF}{GT}$$
   - Trong `scene01`, hệ thống đạt độ chính xác hợp nhất xuất sắc lên đến **0.943** (94.3%).

---

## 4. Bảng Tổng Hợp So Sánh Tính Bổ Trợ Của 3 Loại Dữ Liệu Trong Bài Báo

| Tiêu chí | AIS (`AIS.txt`) | Marine Radar (`radar_images/`) | Visible Camera (`visible_images/`) |
|---|---|---|---|
| **Bản chất dữ liệu** | Bản tin số vô tuyến VHF | Ảnh vệt quét sóng điện từ 2D PPI | Ảnh/video quang học RGB 2K |
| **Tần số cập nhật** | Thấp & không đều (10s - 3 phút) | Trung bình (chu kỳ ~2.24s, 0.45 Hz) | Rất cao (25 Hz / 40 ms) |
| **Không gian tọa độ** | Tọa độ địa lý cầu WGS-84 (Lat, Lon) | Tọa độ cực / pixel ảnh quét BEV ($2048 \times 2048$) | Tọa độ phối cảnh ảnh 2D ($2560 \times 1440$) |
| **Thông tin định danh** | Toàn cầu, chính xác tuyệt đối (MMSI) | ID nội bộ theo vòng quét | ID nội bộ theo mạng Re-ID |
| **Thông tin động học** | SOG, COG trực tiếp từ GPS tàu | Vector vận tốc tính từ vệt quét | Vận tốc biểu kiến trên mặt phẳng ảnh |
| **Khả năng xuyên thời tiết**| Không bị ảnh hưởng bởi thời tiết | Xuyên sương mù, mưa rào, bóng đêm | Bị suy giảm mạnh khi sương mù, mưa, đêm |
| **Nhược điểm lớn nhất** | Tàu nhỏ không trang bị; độ trễ cao | Không có hình dáng thật, nhiễu clutter | Thiếu thông tin chiều sâu (Depth ambiguity) |
| **Đóng góp khi hợp nhất** | Cung cấp MMSI, vị trí GPS, SOG, COG | Cung cấp tọa độ 2D BEV và cổng giám sát Gate | Cung cấp hình ảnh chi tiết, bám vết mượt 25Hz |

---

## 6. Hướng Dẫn Tính Toán Và Hiệu Chuẩn Ma Trận Trên `scene01`

Hệ thống cung cấp sẵn công cụ thực thi hoàn chỉnh tại [`scripts/calibrate_matrices.py`](file:///mnt/d/Workspace/VTX/Fusion/scripts/calibrate_matrices.py) để tự động hóa việc tính toán các ma trận biến đổi tọa độ theo đúng Công thức (10) và Công thức (6) của bài báo.

### 6.1 Lệnh Thực Thi
Chạy lệnh sau trên terminal tại thư mục gốc của dự án:
```bash
python scripts/calibrate_matrices.py --scene datasets/scene01
```

### 6.2 Kết Quả Tính Toán Giải Tích Thực Tế
Công cụ tự động đồng bộ 101 khung quét radar của tàu `413806072` (Radar Track 26 $\leftrightarrow$ Camera Track 3), giải thuật toán hồi quy bình phương tối thiểu và in ra kết quả:
- **Ma trận Affine $M_{geo2rad}$**: Đạt sai số tái chiếu RMSE chỉ **$1.283\text{ pixel}$** (tương đương **$6.26\text{ mét}$** trên mặt nước sông).
- **Ma trận Chiếu $F_{rad2vis}$**: Đạt độ trùng khớp IoU **$59.9\%$** giữa cổng dự báo và thân tàu thực tế, với hệ số chính xác:
  ```python
  F_RADAR_TO_VISIBLE = np.array([
      [-1.38157877e+01, -8.78508291e+00,  3.51290525e+04],
      [ 2.31098562e-01, -2.83131713e-01,  6.87422028e+02],
      [ 2.00502139e+00,  1.84434496e+00, -5.27668883e+03],
      [ 2.13125082e-01,  2.51459832e-01, -6.02300267e+02]
  ], dtype=np.float64)
  ```

---

## 7. Kết Luận

Phân đoạn `datasets/scene01` trong bộ dữ liệu **WHUT-MSFVessel** là một minh chứng tiêu biểu cho tính hiệu quả của phương pháp tiếp cận hợp nhất đa nguồn:
1. **Đầu vào**: Hệ thống thu nhận 3 luồng dữ liệu độc lập, không đồng nhất cả về thời gian (từ 40ms đến hàng phút) và không gian (từ tọa độ cầu GPS WGS-84 đến góc nhìn chim bay BEV của Radar và góc nhìn phối cảnh của Camera).
2. **Xử lý**: Thông qua mô hình **MSTrack** (với vòng lặp phản hồi Re-detection thông minh) và thuật toán **Cascade Association Matching** 4 bước (với hàm chi phí động học kết hợp khoảng cách không gian, độ lệch phương vị và cosine vận tốc), bài báo giải quyết trọn vẹn bài toán trôi ID, mất dấu do sương mù và sai số biến đổi tọa độ.
3. **Đầu ra**: Cung cấp quỹ đạo tàu hợp nhất chuẩn xác (`FusedTrack`), xác định rõ ràng tàu thuộc nhóm Tri-fusion (Đỏ), Dual-fusion (Vàng) hay Single (Xanh), gắn kèm đầy đủ 7 tham số hải đồ cốt lõi, đạt độ chính xác hợp nhất **94.3%** và điểm số **MOTA 0.941** trên dữ liệu thực tế sông Dương Tử.
