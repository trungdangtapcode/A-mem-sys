# Giải thích logic package `agentic_memory`

Tài liệu này không đi theo kiểu "API reference", mà giải thích package theo câu hỏi: package này đang mô hình hóa điều gì trong hệ thống bộ nhớ của A-MEM?

## 1. Package này là phần lõi của repo

Toàn bộ logic đáng quan tâm gần như nằm ở đây:

- [`memory_system.py`](memory_system.py): trung tâm điều phối
- [`llm_controller.py`](llm_controller.py): cầu nối tới các LLM backend
- [`retrievers.py`](retrievers.py): lớp truy hồi vector bằng Chroma

## 2. `MemoryNote` đại diện cho điều gì?

`MemoryNote` là đơn vị ký ức cơ bản.

Điều quan trọng là repo không coi memory chỉ là một đoạn text. Một memory trong package này có ba lớp:

### Lớp 1. Nội dung gốc

- `content`

Đây là thông tin ban đầu mà agent muốn ghi nhớ.

### Lớp 2. Diễn giải ngữ nghĩa

- `keywords`
- `context`
- `tags`

Ba trường này cho phép hệ thống hiểu memory đang nói về điều gì, thuộc cụm chủ đề nào, và nên được tìm lại trong tình huống nào.

### Lớp 3. Quan hệ với các memory khác

- `links`
- `evolution_history`

Đây là phần giúp memory đi từ "một mẩu thông tin riêng lẻ" thành "một nút trong mạng tri thức".

## 3. `AgenticMemorySystem` là gì?

Nếu phải chọn một class đại diện cho cả repo, đó là `AgenticMemorySystem`.

Class này thể hiện toàn bộ vòng đời của một memory:

1. nhận nội dung mới
2. phân tích ý nghĩa
3. tìm memory liên quan
4. quyết định có nên tiến hóa không
5. lưu vào hệ thống
6. cho phép truy hồi lại sau này

## 4. Phần "agentic" nằm ở đâu?

Nằm ở chỗ hệ thống không dừng ở bước lưu và tìm.

Khi một memory mới đi vào hệ thống, `process_memory(...)` sẽ:

- tìm các memory gần nhất
- gửi memory mới cùng các neighbor cho LLM
- hỏi model xem có nên cập nhật cấu trúc bộ nhớ không

Hiện tại trong code, cấu trúc bộ nhớ có thể được điều chỉnh theo hai cách:

- thêm liên kết mới giữa các memory
- cập nhật lại `context` và `tags` của các neighbor

Chính chỗ này làm repo khác với một vector store thông thường.

## 5. Vì sao cần `llm_controller.py`?

Vì phần hiểu ngữ nghĩa và phần quyết định evolution đều phụ thuộc vào model.

Nhưng về mặt kiến trúc, package không muốn gắn cứng vào một nhà cung cấp. Vì thế nó tạo một lớp adapter chung:

- OpenAI
- Ollama
- SGLang
- OpenRouter

Ý nghĩa của file này là:

- logic bộ nhớ giữ nguyên
- backend model có thể thay

Nói cách khác, đây là phần "thay động cơ nhưng giữ nguyên chiếc xe".

## 6. Vì sao cần `retrievers.py`?

Vì muốn memory được tìm lại theo nghĩa, không chỉ theo từ.

Điểm hay trong implementation hiện tại là khi lưu vào Chroma, hệ thống không embed mỗi nội dung thô. Nó tạo một `enhanced_document` gồm:

- nội dung
- context
- keywords
- tags

Hệ quả là retrieval không chỉ dựa vào câu chữ ban đầu, mà còn dựa vào lớp ý nghĩa mà LLM đã sinh ra.

Đây là một chi tiết rất quan trọng nếu bạn đang đọc repo theo logic paper.

## 7. Luồng khái niệm của package

Có thể đọc package này như sơ đồ sau:

```text
Memory thô
  -> LLM diễn giải
  -> Memory giàu metadata
  -> So sánh với memory cũ
  -> Điều chỉnh mạng memory
  -> Lưu để truy hồi lần sau
```

## 8. Điều gì là "nguồn sự thật" trong package?

Về mặt runtime:

- `self.memories` là nơi giữ object `MemoryNote` trong RAM
- Chroma là nơi phục vụ truy hồi semantic

Điều này có nghĩa:

- Chroma giúp tìm nhanh
- còn cấu trúc object sống của hệ thống vẫn nằm nhiều ở `self.memories`

## 9. Nên đọc code theo thứ tự nào?

Nếu muốn hiểu logic package, đọc theo thứ tự này là hợp lý nhất:

1. `MemoryNote` trong [`memory_system.py`](memory_system.py)
2. `add_note(...)` trong [`memory_system.py`](memory_system.py)
3. `process_memory(...)` trong [`memory_system.py`](memory_system.py)
4. `search(...)` và `search_agentic(...)` trong [`memory_system.py`](memory_system.py)
5. `LLMController` trong [`llm_controller.py`](llm_controller.py)
6. `ChromaRetriever` trong [`retrievers.py`](retrievers.py)

## 10. Một câu chốt về package này

`agentic_memory` là phần hiện thực ý tưởng:

> memory không chỉ được lưu trữ, mà còn được hiểu, đặt vào ngữ cảnh, liên kết với ký ức cũ, và có thể được điều chỉnh lại khi tri thức mới xuất hiện.
