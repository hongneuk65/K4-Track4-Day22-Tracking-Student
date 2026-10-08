# Lab Tracking — 2 giờ

Nhóm 2 người một máy. Detector đã khóa. Bạn chọn tracker và ngưỡng cho năm video khác cảnh.

## Việc cần làm

1. Tạo môi trường một lần:

```bash
conda env create -f environment.yml
conda activate cv_robotics_lab21
git clone https://github.com/JonathonLuiten/TrackEval.git
pip install -e TrackEval/
```

2. Tải ảnh năm video: [data_lab21.zip](https://drive.google.com/file/d/1UeVPQd6j5pSzxoJDcKJrerT9SL3vJLDt/view?usp=sharing). Giải nén, rồi gán đường dẫn thư mục chứa `video_1` … `video_5`:

```bash
export LAB_DATA=/đường/dẫn/lab_data
python scripts/check_data.py --lab-data-root "$LAB_DATA"
```

Cả năm video phải có ảnh. Chỉ `video_1` có nhãn.

3. Mở `on_tap_metrics.ipynb` bằng kernel env này. Đọc bảng MOTA / IDF1 / HOTA, rồi chạy YOLO trên một ảnh `video_1`.

4. Chạy tracker. Bản thử có thể giới hạn frame. Bản nộp thì không.

```bash
python scripts/run_tracking.py \
  --source "$LAB_DATA/video_1/img1" \
  --seq-name video_1 \
  --tracker bytetrack --conf 0.3 --iou 0.5 \
  --out runs/nop_bai --save-video
```

Đổi `--seq-name` và thư mục `img1` cho `video_2` … `video_5`. Tracker được chọn: `bytetrack`, `ocsort`, `botsort`, `strongsort`, `deepocsort`.

5. Chấm số **chỉ** `video_1`:

```bash
python scripts/evaluate_practice.py \
  --trackeval-root ~/TrackEval \
  --lab-data-root "$LAB_DATA" \
  --submission runs/nop_bai/video_1.txt \
  --run-name nhom01_video1
```

`video_2` đến `video_5` không có nhãn. Xem `preview/video_N.mp4` và video có vẽ ID, rồi ghi điều bạn thấy.

## Luật chơi

| Khóa | Bạn chọn |
|---|---|
| Detector `yolo26n.pt`, ảnh 640 px, lớp người, Re-ID `osnet_x0_25_msmt17.pt` | Tracker, `--conf`, `--iou` của detector |

## Nộp

- `video_1.txt` … `video_5.txt` trong `runs/nop_bai/` (đủ frame, đúng tên).
- `submission_template/BAO_CAO_mau.md` đã điền. Số HOTA / MOTA / IDF1 chỉ bắt buộc cho `video_1`.

Bản đã hoàn thiện có đủ năm file TXT tại `runs/nop_bai/`, cấu hình từng video,
`kiem_tra.json` và `video_1_metrics.json`. Báo cáo nằm tại
`submission_template/BAO_CAO_mau.md` (bản sao: `BAO_CAO_hoan_thien.md`).
Cần bổ sung tên nhóm và thành viên trước khi nộp chính thức.

Chi tiết từng bước, sự cố, và lịch 2 giờ: [HUONG_DAN.md](HUONG_DAN.md).

Trên máy Windows đã thiết lập môi trường, có thể kích hoạt nhanh từ thư mục repo:

```powershell
. .\scripts\activate_lab.ps1
```

Script đặt UTF-8 và tự nhận thư mục `data_lab21` cạnh repo nếu chưa có `LAB_DATA`.
Nếu cài mới bằng Python 3.11, xem hướng dẫn `requirements-windows.txt` trong
`HUONG_DAN.md` để tránh pin NumPy cũ của BoxMOT.
