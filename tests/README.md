# Giải thích thư mục `tests`

Thư mục này không phải phần logic chính của paper, nhưng nó giúp nhìn nhanh repo đang muốn giữ ổn định những hành vi nào.

## 1. `test_memory_system.py`

File này kiểm tra các hành vi cốt lõi của hệ thống bộ nhớ:

- tạo memory mới
- giữ được metadata sau khi lưu và tìm lại
- cập nhật memory
- xóa memory
- truy hồi memory liên quan
- xử lý liên kết giữa các memory
- chạy luồng evolution ở mức cơ bản

Nói ngắn gọn: file này bảo vệ "hệ thống bộ nhớ có chạy đúng vòng đời chính hay không".

## 2. `test_llm_backends.py`

File này tập trung vào phần backend model, đặc biệt là SGLang:

- khởi tạo backend đúng chưa
- payload gửi đi có đúng format chưa
- nếu lỗi mạng hoặc lỗi server thì có fallback hợp lý không

Nói ngắn gọn: file này bảo vệ "phần cầu nối sang LLM có đúng giao thức không".

## 3. `test_utils.py`

File này chứa mock đơn giản để phục vụ test.

## 4. Cách hiểu vai trò của test trong repo này

Nếu đọc repo theo tinh thần paper, test ở đây có ý nghĩa:

- xác nhận memory có thể được tạo và truy hồi
- xác nhận metadata có đi xuyên qua hệ thống
- xác nhận cơ chế evolution không làm vỡ flow chính
- xác nhận các backend LLM có thể được thay mà không đổi logic lõi

## 5. Trạng thái chạy test trong môi trường hiện tại

Mình đã thử chạy:

```bash
python3 -m unittest
```

nhưng môi trường hiện tại đang thiếu package `litellm`, nên test chưa thể chạy hết ngay trong phiên này.
