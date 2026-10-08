# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** Chưa cung cấp — bổ sung trước khi nộp.

**Thành viên:** Chưa cung cấp — bổ sung trước khi nộp.

Detector cố định: `yolo26n.pt`, ảnh 640 px, lớp người; Re-ID của BoT-SORT: `osnet_x0_25_msmt17.pt`. Giữ nguyên tham số nội bộ của tracker. Chạy trên CPU trong môi trường `cv_robotics_lab21` (Python 3.11, BoxMOT 10.0.42, NumPy 1.26.4, PyTorch 2.5.1).

## 1. Cấu hình đã chọn

| Video | Tracker | conf | iou | Quan sát | Cấu hình đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 | BoT-SORT | 0.15 | 0.5 | Hai người gần camera giữ ID qua các frame mẫu; ngưỡng thấp giữ thêm người ở xa. Điểm tổng hợp trên đủ 600 frame tốt hơn cấu hình ByteTrack đã chấm. | ByteTrack, conf=0.3, iou=0.5: nhanh hơn và ít hộp sai hơn nhưng HOTA thấp hơn. |
| video_2 | BoT-SORT | 0.15 | 0.5 | Cảnh đêm đông người; ngưỡng thấp giữ thêm một số người, nhiều người nhỏ ở xa vẫn bị bỏ sót. | BoT-SORT conf=0.5 bỏ nhiều người nhìn thấy; ByteTrack conf=0.3 cũng bỏ một số người nhỏ. |
| video_3 | ByteTrack | 0.3 | 0.5 | Hai người nổi bật ở tiền cảnh giữ ID qua frame 1, 75, 150 với cả hai tracker; ByteTrack chạy nhanh hơn. | BoT-SORT conf=0.3 chưa thể hiện lợi ích rõ ở các mẫu; ByteTrack conf=0.15 mất track người áo sọc tại frame 50. |
| video_4 | ByteTrack | 0.3 | 0.5 | Người áo đỏ và áo trắng giữ ID qua các frame mẫu trong cảnh camera di chuyển; ByteTrack đủ cho những đoạn này. | BoT-SORT conf=0.3 chậm hơn, thêm vài hộp nền nhỏ nhưng chưa có lợi ích giữ ID rõ ở mẫu quan sát. |
| video_5 | BoT-SORT | 0.3 | 0.5 | Giữ thêm một số người nhỏ bên đường so với ByteTrack; người nhỏ, tối và bị che vẫn có track đứt hoặc đổi ID. | BoT-SORT conf=0.5 mất nhiều hộp; conf=0.15 không cải thiện nhất quán giữa frame 50 và 100. |

Đã thử cả hai tracker trên 150 frame đầu của từng video với conf=0.3, iou=0.5. Sau đó thử conf=0.15 và 0.5 trên 100 frame, iou=0.4 và 0.7 trên 50 frame với tracker dự kiến chọn. Mẫu thử IoU cho kết quả giống nhau ở một số cảnh, khác ở video_1; kiểm tra phát hiện riêng cũng thấy khác tại một số frame của video_3. Giữ iou=0.5 vì chưa có bằng chứng đủ mạnh để đổi cho bản nộp.

Bản nộp chạy toàn bộ 600 / 1050 / 837 / 900 / 750 frame tương ứng video_1 đến video_5, tổng 4137 frame, không giới hạn số frame. Trên máy thực nghiệm, các ảnh `runs/nop_bai/minh_chung/video_N_toan_bo.jpg` minh họa đầu, giữa và cuối mỗi video; ảnh so sánh tracker và ngưỡng nằm trong `runs/quan_sat`. Các ảnh này là minh chứng cục bộ, không thuộc phần kết quả đẩy lên GitHub.

## 2. Số liệu video_1

Chấm bằng `scripts/evaluate_practice.py` với nhãn video_1 và đủ 600 frame. Các số dưới đây lấy từ kết quả TrackEval, theo thang 0–100.

```text
Cấu hình                     HOTA       MOTA       IDF1
ByteTrack conf=0.30 iou=0.50   26.912     17.292     25.713
BoT-SORT  conf=0.15 iou=0.50   29.343     20.731     29.561
```

| Cấu hình | TP | FN | FP | Đổi ID | AssA | Tốc độ xử lý tham khảo |
|---|---:|---:|---:|---:|---:|---:|
| ByteTrack conf=0.3 | 3332 | 15249 | 107 | 12 | 48.130 | 7.1 frame/giây |
| BoT-SORT conf=0.15 | 4384 | 14197 | 505 | 27 | 45.113 | 3.8 frame/giây |

Chọn BoT-SORT conf=0.15 vì HOTA, MOTA và IDF1 cao hơn trong hai cấu hình được chấm. Tuy nhiên FP và số lần đổi ID tăng, AssA giảm: không kết luận BoT-SORT tốt hơn ở mọi mặt. Hai cấu hình đồng thời khác tracker và conf, vì vậy chênh lệch này là so sánh cấu hình, không chứng minh riêng tác dụng của Re-ID. Tốc độ đo trên CPU trong lần chạy này, gồm xử lý và xuất preview, chỉ dùng tham khảo.

`video_2` đến `video_5` không có nhãn trong gói lab; chỉ đánh giá bằng mắt, không tính hoặc suy diễn HOTA/MOTA/IDF1 cho chúng. Kết quả chấm bản chọn nằm ở `runs/nop_bai/video_1_metrics.json`; cấu hình từng video và kết quả kiểm tra định dạng cũng được lưu cùng năm file TXT.

## 3. Phân tích

### video_1

Cảnh ban ngày và camera tĩnh giúp các người lớn ở tiền cảnh được phát hiện khá đều. Trong các frame mẫu, cả hai tracker giữ được ID của hai người đi gần camera, còn BoT-SORT giữ thêm người nhỏ ở phía xa. Trên đủ video, cấu hình BoT-SORT ngưỡng thấp tăng TP từ 3332 lên 4384 và tăng HOTA từ 26.912 lên 29.343. Đổi lại FP tăng từ 107 lên 505 và đổi ID từ 12 lên 27, nên lựa chọn này ưu tiên điểm tổng hợp và giảm bỏ sót chứ không giải quyết hoàn toàn lỗi danh tính. Số FN vẫn rất lớn, cho thấy giới hạn phát hiện người nhỏ với detector nano và ảnh 640.

### video_2

Cảnh tối và đông làm nhiều người xa camera chỉ chiếm vùng ảnh nhỏ hoặc bị che. Ở frame 50, BoT-SORT xuất 12, 11 và 7 hộp với conf lần lượt 0.15, 0.3 và 0.5; số hộp không phải số người đúng. Quan sát hình cho thấy ngưỡng cao bỏ thêm người nhìn thấy, nên chọn 0.15 để giữ thêm ứng viên cho tracker. Người gần camera tương đối dễ theo dõi, nhưng các nhóm xa vẫn thường thiếu hộp, do đó không khẳng định kết quả đã đầy đủ.

### video_3

Camera di chuyển và ảnh nhỏ làm vị trí người thay đổi nhanh. Hai người nổi bật ở tiền cảnh vẫn giữ ID qua frame 1, 75 và 150 với cả ByteTrack và BoT-SORT trong thử nghiệm. Chưa thấy lợi ích rõ của Re-ID trong các đoạn mẫu này, trong khi ByteTrack đạt khoảng 10.7 frame/giây và BoT-SORT khoảng 7.1 frame/giây ở lần thử 150 frame. Chọn ByteTrack conf=0.3 vì cân bằng theo dõi và tốc độ; conf=0.15 không luôn tăng số track hữu ích, như người áo sọc bị thiếu ở frame 50.

### video_4

Cảnh trong nhà có camera tiến về phía trước, nhiều người nhỏ ở nền và bề mặt phản chiếu. Cả hai tracker giữ ID người áo đỏ và áo trắng qua các frame mẫu. BoT-SORT thêm một số hộp nhỏ ở nền nhưng chưa cho thấy cải thiện danh tính đủ rõ để bù tốc độ thấp hơn. Chọn ByteTrack conf=0.3; vẫn cần kiểm tra các đoạn che khuất dài vì mẫu đầu video không bao quát mọi lỗi.

### video_5

Camera trên xe khiến nền và vị trí người bên đường thay đổi mạnh. Ở mẫu so sánh, BoT-SORT giữ thêm một số người nhỏ so với ByteTrack, nhưng vùng đông và người tối vẫn có track đứt hoặc đổi ID. conf=0.5 bỏ nhiều hộp nhìn thấy, còn conf=0.15 không cải thiện nhất quán giữa hai frame kiểm tra. Chọn BoT-SORT conf=0.3 như phương án cân bằng, không suy diễn chất lượng định lượng khi không có nhãn.

## 4. Nếu có thêm thời gian

Kiểm tra bổ sung bản chạy đủ frame: ở video_1 frame 300, vùng hai người bên phải có các hộp chồng nhau; frame 600 vẫn có nhiều người nền thiếu hộp. video_2 frame 525 và 1050 giữ được nhiều người gần camera nhưng bỏ nhiều người ở vùng đông phía trên trái. video_3 frame 418 và 837 có hộp quanh nhiều người lớn, nhưng người nhỏ ở nền vẫn thiếu. Trong video_4, người áo đỏ có ID 204 ở frame 450 và ID 238 ở frame 900, cho thấy bản chạy dài vẫn có thay đổi ID dù các mẫu đầu giữ ổn định. Các ảnh tĩnh này không đủ để kết luận ID liên tục giữa các thời điểm cách xa nhau.

Quét conf mịn hơn trên đủ video_1 và kiểm tra các đoạn tạo FP, đổi ID hoặc mất người ở xa. Với các video không nhãn, xem thêm đoạn che khuất và giao cắt trước khi quyết định đổi cấu hình.

Ở video_5, frame 375 có hộp quanh một số người đang băng đường bên phải nhưng nhóm phía sau vẫn thiếu; frame 750 còn nhiều người nhỏ hoặc trong bóng tối không có hộp. Điều này phù hợp với hạn chế phát hiện đã thấy ở mẫu đầu video.
