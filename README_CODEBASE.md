# Tài liệu giải thích logic A-MEM

Tài liệu này được viết để đọc repo theo góc nhìn ý tưởng của paper: hệ thống bộ nhớ hoạt động ra sao, vì sao có các bước đó, và mỗi phần code đang đại diện cho phần nào trong logic của A-MEM.

## 1. Repo này đang hiện thực điều gì?

Nếu nói ngắn gọn, repo này hiện thực một ý tưởng rất đơn giản:

> Thay vì chỉ nhét mọi ký ức vào một kho vector rồi tìm lại bằng similarity search, agent nên có khả năng "chủ động tổ chức lại ký ức" của mình.

Trong repo này, việc "tổ chức lại ký ức" được thể hiện qua 3 việc:

- Tự phân tích một memory mới để sinh ra metadata có ý nghĩa
- Tìm các memory gần nhất để hiểu memory mới đang liên quan đến cụm kiến thức nào
- Quyết định có nên làm giàu memory mới hoặc cập nhật các memory lân cận hay không

Nói cách khác, đây không chỉ là hệ thống `lưu` và `tìm`, mà là hệ thống `lưu -> hiểu -> liên kết -> điều chỉnh`.

## 2. Ý tưởng cốt lõi của A-MEM trong repo

Repo này xoay quanh một vòng lặp:

```text
Thông tin mới
  -> Phân tích ngữ nghĩa
  -> Lưu vào bộ nhớ
  -> So với ký ức cũ
  -> Tạo liên kết / cập nhật ngữ cảnh
  -> Dùng lại khi truy hồi
```

Paper gọi đây là hướng tiếp cận "agentic memory", còn trong code thì nó đang được cụ thể hóa bằng lớp `AgenticMemorySystem`.

## 3. Ba khối chính của hệ thống

### 3.1. Khối điều phối bộ nhớ

File: [`agentic_memory/memory_system.py`](agentic_memory/memory_system.py)

Đây là phần quan trọng nhất. Nó giữ vai trò:

- nhận memory mới
- quyết định có cần phân tích thêm không
- gọi LLM để sinh metadata
- tìm các memory liên quan
- quyết định có "evolve" không
- lưu memory vào cấu trúc đang chạy

Bạn có thể hiểu lớp này như "bộ não điều phối" của cả hệ thống.

### 3.2. Khối gọi mô hình ngôn ngữ

File: [`agentic_memory/llm_controller.py`](agentic_memory/llm_controller.py)

Phần này không chứa logic bộ nhớ, mà chỉ là lớp trung gian để hỏi model:

- OpenAI
- Ollama
- SGLang
- OpenRouter

Vai trò của nó là: hệ thống bộ nhớ cần một nơi thống nhất để lấy phân tích ngữ nghĩa, còn model cụ thể nào thì có thể thay đổi.

### 3.3. Khối truy hồi bằng vector

File: [`agentic_memory/retrievers.py`](agentic_memory/retrievers.py)

Phần này làm hai việc:

- lưu memory vào ChromaDB
- truy tìm các memory gần với câu truy vấn

Điểm quan trọng là repo không embed mỗi `content`, mà ghép thêm:

- `context`
- `keywords`
- `tags`

Điều đó phản ánh đúng tinh thần paper: trí nhớ không chỉ là văn bản thô, mà là văn bản đã được gắn nghĩa.

## 4. Một memory trong repo gồm những gì?

Trong code, một đơn vị ký ức là `MemoryNote`.

Nó gồm các trường chính:

- `content`: nội dung gốc
- `keywords`: từ khóa quan trọng
- `context`: ngữ cảnh, chủ đề hoặc ý nghĩa tổng quát
- `tags`: các nhãn phân loại
- `links`: liên kết đến các memory khác
- `timestamp`: thời điểm tạo
- `retrieval_count`: số lần được gọi lại
- `evolution_history`: lịch sử tiến hóa

Nhìn theo logic của paper:

- `content` là dữ liệu gốc
- `keywords/context/tags` là lớp diễn giải ngữ nghĩa
- `links` là cấu trúc mạng lưới tri thức giữa các memory

## 5. Luồng hoạt động quan trọng nhất: thêm một memory mới

Đây là luồng quan trọng nhất để hiểu repo.

### Bước 1. Tạo `MemoryNote`

Khi gọi `add_note(...)`, hệ thống tạo một object memory mới từ nội dung đầu vào.

### Bước 2. Phân tích ngữ nghĩa nếu metadata còn thiếu

Nếu memory chưa có:

- `keywords`
- `context`
- `tags`

thì hàm `analyze_content(...)` sẽ gọi LLM để sinh ra các trường này.

Ý nghĩa về mặt paper:

- agent không chỉ ghi nhớ "nguyên văn"
- agent cố hiểu memory mới đang nói về cái gì
- metadata này giúp truy hồi và tổ chức trí nhớ tốt hơn về sau

### Bước 3. So memory mới với các memory cũ

Sau khi có metadata, hệ thống gọi `process_memory(...)`.

Trong đó:

- memory mới được dùng làm truy vấn
- hệ thống tìm các memory gần nhất qua Chroma
- các memory gần nhất này được đưa vào prompt cho LLM

Mục tiêu là để model trả lời câu hỏi:

- memory mới có nên được nối với memory nào không?
- có nên cập nhật lại bối cảnh hoặc tags của các memory lân cận không?

### Bước 4. Thực hiện "evolution"

Nếu LLM quyết định `should_evolve = true`, hiện tại repo hỗ trợ hai kiểu hành động:

- `strengthen`: tăng liên kết giữa memory mới và các memory liên quan
- `update_neighbor`: cập nhật `context` và `tags` của các memory lân cận

Đây là phần gần nhất với ý tưởng "agentic" trong paper.

Hệ thống không chỉ nói:

> "memory này giống memory kia"

mà còn nói:

> "vì chúng liên quan, ta nên làm lại cấu trúc bộ nhớ một chút"

### Bước 5. Lưu memory

Sau khi xử lý xong:

- memory được đưa vào `self.memories`
- đồng thời được đưa vào ChromaDB để phục vụ truy hồi

## 6. Khi tìm kiếm, hệ thống trả về cái gì?

Repo có hai kiểu tìm chính.

### `search(...)`

Đây là kiểu tìm gọn hơn:

- query Chroma
- lấy ra các memory tương ứng
- trả về nội dung, context, keywords, tags, score

Đây là truy hồi semantic cơ bản.

### `search_agentic(...)`

Đây là phiên bản gần với "memory network" hơn:

- truy hồi các memory phù hợp
- cố kéo theo các neighbor từ `links`
- trả về dữ liệu giàu metadata hơn

Nó cho thấy hệ thống không muốn xem mỗi memory là một điểm rời rạc, mà muốn xem memory như một mạng liên kết.

## 7. Repo này đang bám paper ở điểm nào?

Nếu nhìn ở mức khái niệm, repo bám paper ở các ý sau:

- Memory không chỉ là raw text
- Memory có thêm lớp diễn giải ngữ nghĩa
- Memory mới được đánh giá trong tương quan với memory cũ
- Hệ thống có khả năng tự cập nhật cấu trúc bộ nhớ
- Truy hồi không chỉ dựa vào nội dung nguyên văn mà còn dựa vào metadata và liên kết

## 8. Repo này chưa hiện thực toàn bộ paper ở mức nào?

Đây là điểm quan trọng để đọc repo cho đúng kỳ vọng.

Repo này là một thư viện memory system, không phải toàn bộ pipeline thí nghiệm của paper.

Vì vậy, nó chủ yếu tập trung vào phần:

- tạo memory
- enrich metadata
- retrieval
- linking
- evolution ở mức đơn giản

Nó không phải là:

- toàn bộ benchmark của paper
- toàn bộ framework agent hoàn chỉnh
- hệ thống production-ready với persistence mạnh

Nói dễ hiểu: đây là "lõi ý tưởng bộ nhớ" của paper được đóng gói thành thư viện dùng lại.

## 9. Cách map từ ý tưởng sang file code

| Ý tưởng trong logic hệ thống | File chính |
|-----------------------------|-----------|
| Một ký ức là gì | `agentic_memory/memory_system.py` (`MemoryNote`) |
| Điều phối vòng đời memory | `agentic_memory/memory_system.py` (`AgenticMemorySystem`) |
| Hỏi LLM để phân tích memory | `agentic_memory/llm_controller.py` |
| Lưu và truy hồi semantic | `agentic_memory/retrievers.py` |
| Kiểm chứng hành vi chính | `tests/test_memory_system.py` |

## 10. Nếu chỉ đọc 3 phần để hiểu repo

Nên đọc theo thứ tự này:

1. [`agentic_memory/README.md`](agentic_memory/README.md)
2. [`agentic_memory/memory_system.py`](agentic_memory/memory_system.py)
3. [`README.md`](README.md)

## 11. Một câu tóm tắt cuối

Repo này hiện thực A-MEM theo cách thực dụng:

> dùng LLM để biến một memory thô thành memory có nghĩa, dùng vector retrieval để đặt memory đó vào đúng vùng tri thức, rồi cho phép hệ thống tự tăng liên kết hoặc điều chỉnh ngữ cảnh của các memory xung quanh.
