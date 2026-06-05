# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Võ Tấn Trung
**Nhóm:** B4
**Ngày:** 5/6/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector embedding có hướng gần giống nhau trong không gian vector. Với text embeddings, điều này thường cho thấy hai câu/chunk có ý nghĩa hoặc chủ đề gần nhau, dù có thể dùng từ ngữ khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: VinWonders Phú Quốc có các khu vui chơi và công viên chủ đề.
- Sentence B: Khu vui chơi VinWonders ở Phú Quốc cung cấp nhiều hoạt động giải trí.
- Tại sao tương đồng: Hai câu cùng nói về địa điểm VinWonders Phú Quốc và các hoạt động vui chơi giải trí.

**Ví dụ LOW similarity:**
- Sentence A: VinWonders Phú Quốc có các khu vui chơi và công viên chủ đề.
- Sentence B: Python là một ngôn ngữ lập trình phổ biến cho phân tích dữ liệu.
- Tại sao khác: Hai câu thuộc hai chủ đề khác nhau hoàn toàn: du lịch/giải trí và lập trình.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào hướng của vector nên phù hợp để so sánh ý nghĩa giữa các văn bản, ít bị ảnh hưởng bởi độ lớn vector. Với text embeddings, hướng thường quan trọng hơn khoảng cách tuyệt đối vì hai đoạn văn có độ dài khác nhau vẫn có thể cùng ý nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap)) = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23`
> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100, số chunk là `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25`, tức là tăng từ 23 lên 25 chunks. Muốn overlap nhiều hơn để giữ thêm ngữ cảnh giữa các chunk liền kề, giúp retrieval ít bị mất thông tin ở ranh giới chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Du lịch

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain du lịch vì nội dung dễ hiểu, gần gũi với người dùng và dễ thu thập dữ liệu từ các trang giới thiệu dịch vụ. Ngoài ra, hiện tại hệ sinh thái Vinpearl chưa có chatbot hỗ trợ người dùng thật tốt, nên bài toán retrieval có ý nghĩa thực tế: giúp người dùng hỏi nhanh về địa điểm, gói dịch vụ, điều khoản, chính sách hoàn huỷ và cách sử dụng voucher.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Aquafield Nha Trang - Spa & xông hơi chuẩn Hàn | Vinpearl website | 7070 | `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index`, `source`, `extension`, `doc_id` |
| 2 | Aquafield Ocean City Hà Nội - Tổ hợp Spa Xông hơi cao cấp nhất Việt Nam | Vinpearl website | 4756 | `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index`, `source`, `extension`, `doc_id` |
| 3 | Grand World | Vinpearl website | 10288 | `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index`, `source`, `extension`, `doc_id` |
| 4 | Vinpearl Safari Phú Quốc | Vinpearl website | 4666 | `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index`, `source`, `extension`, `doc_id` |
| 5 | VinWonders Nha Trang | Vinpearl website | 6196 | `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index`, `source`, `extension`, `doc_id` |
| 6 | VinWonders Phú Quốc | Vinpearl website | 7029 | `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index`, `source`, `extension`, `doc_id` |
| 7 | [Cần Thơ] 2N1Đ phòng Deluxe + Bữa sáng tại Vinpearl Hotel Cần Thơ | Vinpearl website | 4993 | `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index`, `source`, `extension`, `doc_id` |
| 8 | [Grand World Phú Quốc] Vé Bảo Tàng Gấu Teddy Bear | Vinpearl website | 3223 | `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index`, `source`, `extension`, `doc_id` |
| 9 | [HCM-Nha Trang] ROAM 2022: Combo 3N2Đ Vinpearl + VMB Vietnam Airlines khứ hồi + Ăn sáng hoặc ăn 3 bữa mỗi ngày | Vinpearl website | 10801 | `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index`, `source`, `extension`, `doc_id` |
| 10 | [Vinpearl Golf Phú Quốc] - Voucher Tee time giá siêu ưu đãi | Vinpearl website | 2845 | `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index`, `source`, `extension`, `doc_id` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `title` | string | `VinWonders Phú Quốc` | Giúp biết chunk thuộc sản phẩm/dịch vụ nào và hiển thị nguồn rõ hơn trong câu trả lời. |
| `url` | string | `https://booking.vinpearl.com/...` | Cho phép truy ngược về trang gốc để kiểm chứng thông tin. |
| `original_price` | string | `1.200.000 đ` | Hữu ích khi người dùng hỏi về giá gốc hoặc so sánh khuyến mãi. |
| `current_price` | string | `950.000 đ` | Hữu ích khi truy vấn về giá hiện tại hoặc ưu đãi. |
| `section` | string | `Điều khoản`, `Chính sách hoàn huỷ` | Giúp lọc/ngữ cảnh hoá câu trả lời theo loại thông tin người dùng cần. |
| `chunk_index` | int | `3` | Giúp định vị chunk trong tài liệu và debug kết quả retrieval. |
| `source` | string | `data/dataset/VinWonders_Phú_Quốc.md` | Giúp biết file nguồn của chunk. |
| `doc_id` | string | `VinWonders_Phú_Quốc` | Dùng để nhóm hoặc xoá toàn bộ các chunk thuộc cùng một tài liệu. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| VinWonders Phú Quốc | FixedSizeChunker (`fixed_size`) | 14 | 497.4 | Trung bình - giữ đúng kích thước nhưng có thể cắt ngang câu/section |
| VinWonders Phú Quốc | SentenceChunker (`by_sentences`) | 15 | 460.3 | Khá tốt - giữ câu hoàn chỉnh nhưng không tận dụng cấu trúc Markdown |
| VinWonders Phú Quốc | RecursiveChunker (`recursive`) | 19 | 363.9 | Tốt - ưu tiên tách theo đoạn, dòng và câu |
| Grand World | FixedSizeChunker (`fixed_size`) | 21 | 486.3 | Trung bình - ổn về kích thước nhưng dễ mất ranh giới ngữ nghĩa |
| Grand World | SentenceChunker (`by_sentences`) | 20 | 507.6 | Khá tốt - nội dung câu dễ đọc nhưng vài chunk hơi dài |
| Grand World | RecursiveChunker (`recursive`) | 27 | 376.8 | Tốt - chunk nhỏ hơn và bám cấu trúc văn bản hơn |
| VinWonders Nha Trang | FixedSizeChunker (`fixed_size`) | 13 | 473.0 | Trung bình - đơn giản nhưng có thể cắt ngang thông tin |
| VinWonders Nha Trang | SentenceChunker (`by_sentences`) | 15 | 407.7 | Khá tốt - giữ câu tự nhiên |
| VinWonders Nha Trang | RecursiveChunker (`recursive`) | 19 | 322.1 | Tốt - cân bằng giữa độ dài và ngữ cảnh |

### Strategy Của Tôi

**Loại:** Custom strategy - Sliding Window Chunking với overlap (`chunk_size=1000`, `overlap=100`)

**Mô tả cách hoạt động:**
> Strategy của em xử lý tài liệu Markdown theo section trước, sau đó áp dụng sliding window trên nội dung từng section. Mỗi chunk có tối đa 1000 ký tự và chunk kế tiếp lặp lại 100 ký tự cuối của chunk trước để giữ ngữ cảnh ở vùng ranh giới. Cách này không chỉ cắt theo kích thước cố định toàn file mà vẫn giữ metadata section như `Mô tả`, `Điều khoản`, `Chính sách hoàn huỷ`, `Hướng dẫn sử dụng`. Nhờ vậy, khi người dùng hỏi về một loại thông tin cụ thể, chunk trả về thường vẫn nằm trong đúng phần nội dung của tài liệu.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Domain du lịch có nhiều thông tin chi tiết nằm trong cùng một mục, ví dụ giá, điều khoản, chính sách hoàn huỷ và hướng dẫn sử dụng voucher. Em chọn chunk lớn hơn baseline vì câu trả lời du lịch thường cần đủ bối cảnh để tránh trả lời thiếu điều kiện hoặc thiếu lưu ý. Overlap 100 ký tự giúp giảm rủi ro mất ý ở đoạn chuyển giữa hai chunk.

**Code snippet (nếu custom):**
```python
def sliding_window_chunk(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    cleaned_text = normalize_whitespace(text)
    if not cleaned_text:
        return []
    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(cleaned_text), step):
        chunk = cleaned_text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(cleaned_text):
            break
    return chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| VinWonders Phú Quốc | best baseline: RecursiveChunker | 19 | 363.9 | Chunk ngắn, bám ranh giới ngữ nghĩa tốt nhưng đôi khi thiếu bối cảnh rộng |
| VinWonders Phú Quốc | **của tôi: Sliding Window 1000/100** | 10 | 718.2 | Giữ nhiều bối cảnh hơn, phù hợp câu hỏi cần điều kiện/chi tiết |
| Grand World | best baseline: RecursiveChunker | 27 | 376.8 | Tách nhỏ và rõ ý, nhưng số chunk nhiều hơn |
| Grand World | **của tôi: Sliding Window 1000/100** | 13 | 838.6 | Ít chunk hơn, mỗi chunk chứa nhiều thông tin liên quan hơn |
| VinWonders Nha Trang | best baseline: RecursiveChunker | 19 | 322.1 | Dễ retrieve ý nhỏ nhưng có thể thiếu phần giải thích đi kèm |
| VinWonders Nha Trang | **của tôi: Sliding Window 1000/100** | 8 | 799.4 | Tốt cho câu hỏi người dùng cần câu trả lời đầy đủ hơn |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Sliding Window 1000/100 | Chờ benchmark | Giữ nhiều bối cảnh trong mỗi chunk, phù hợp câu hỏi về điều khoản/gói dịch vụ | Chunk dài hơn nên có thể kéo theo thông tin phụ không cần thiết |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Sau khi nhóm thử nghiệm nhiều phương pháp, SectionChunker là strategy phù hợp nhất cho domain du lịch. Lý do là các tài liệu Vinpearl/VinWonders có cấu trúc rõ theo section như `Mô tả`, `Điều khoản`, `Chính sách hoàn huỷ`, `Hướng dẫn sử dụng`, nên tách theo section giúp chunk giữ đúng ngữ cảnh nghiệp vụ. Cách này cũng giúp retrieval trả về đúng loại thông tin người dùng hỏi hơn so với việc chỉ cắt theo độ dài ký tự.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Em dùng regex `(?<=[.!?])\s+|\.\n` để tách văn bản tại ranh giới câu sau dấu chấm, chấm than, chấm hỏi hoặc dấu chấm trước xuống dòng. Sau khi tách, em strip khoảng trắng và bỏ các câu rỗng, rồi gom tối đa `max_sentences_per_chunk` câu vào một chunk để chunk vẫn dễ đọc và không bị quá vụn.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `RecursiveChunker` kiểm tra nếu text đã ngắn hơn `chunk_size` thì trả về luôn; nếu quá dài thì thử tách theo thứ tự separator ưu tiên như đoạn, dòng, câu, khoảng trắng. Với mỗi phần vẫn quá dài, hàm `_split` gọi đệ quy với separator tiếp theo; nếu hết separator thì fallback sang `FixedSizeChunker` để đảm bảo luôn tạo được chunk.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` tạo embedding cho từng `Document`, lưu content, metadata, doc_id và vector embedding vào in-memory store; nếu ChromaDB dùng được thì cũng add vào collection. `search` embed query, tính điểm giữa query embedding và từng document embedding bằng dot product, sau đó sort giảm dần và trả về top-k kết quả.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` filter metadata trước, chỉ giữ các record khớp toàn bộ key-value trong `metadata_filter`, rồi mới chạy similarity search trên tập ứng viên nhỏ hơn. `delete_document` tìm các chunk có `doc_id` tương ứng, xoá khỏi in-memory store và nếu ChromaDB đang hoạt động thì xoá các id đó khỏi collection.

### KnowledgeBaseAgent

**`answer`** — approach:
> `KnowledgeBaseAgent.answer` retrieve top-k chunks từ vector store, sau đó dựng context gồm title, source, url, giá, section, chunk index và nội dung chunk. Prompt yêu cầu LLM chỉ trả lời dựa trên context, luôn trả lời bằng tiếng Việt, và nói không biết nếu context không đủ thông tin.

### Test Results

```
# Paste output of: pytest tests/ -v
Ghi chú: môi trường Codex hiện không có module pytest, nên em kiểm tra tương đương bằng unittest:

Ran 42 tests in 0.011s
OK
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | VinWonders Phú Quốc có công viên nước và nhiều trò chơi giải trí. | VinWonders Phú Quốc là khu vui chơi có nhiều hoạt động giải trí cho du khách. | high | 0.8106 | Đúng |
| 2 | Vé Vinpearl Safari Phú Quốc có giá hiện tại là 850.000 đồng. | Giá vé Safari Phú Quốc hiện tại là 850.000 đồng. | high | 0.9566 | Đúng |
| 3 | Aquafield Nha Trang là spa xông hơi chuẩn Hàn. | Python là ngôn ngữ lập trình dùng cho trí tuệ nhân tạo. | low | 0.5414 | Tương đối đúng |
| 4 | Khách hàng có thể xuất trình voucher điện tử tại cổng. | Du khách có thể dùng voucher online để vào cửa. | high | 0.7681 | Đúng |
| 5 | VinWonders Phú Quốc có xe buýt đưa đón từ Dương Đông. | Bảo tàng Teddy Bear có hơn 500 chú gấu trưng bày. | low | 0.4240 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là pair 3 vẫn có score 0.5414 dù hai câu thuộc hai chủ đề khác nhau. Điều này cho thấy embedding không chỉ so sánh keyword trực tiếp mà biểu diễn câu trong một không gian ngữ nghĩa liên tục, nên các câu có cấu trúc chung hoặc cùng kiểu mô tả vẫn có thể nhận điểm dương. Tuy nhiên, các cặp thật sự cùng ý như pair 2 vẫn có điểm cao hơn rõ rệt.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Điều hay nhất em học được từ thành viên khác là cách chọn metadata sao cho có ích cho retrieval, không chỉ lưu thông tin cho đủ. Ví dụ các trường như `section`, `title`, `current_price`, `doc_id` giúp hệ thống trả lời đúng ngữ cảnh hơn. Em cũng học được cách đặt benchmark query sao cho có thể kiểm chứng bằng tài liệu gốc và bao phủ nhiều loại câu hỏi khác nhau.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Điều hay nhất em học được từ nhóm khác là có thể kết hợp nhiều phương pháp chunking để cải thiện dữ liệu đầu vào cho LLM. Thay vì chỉ dùng một cách cắt cố định, có thể tận dụng cấu trúc tài liệu, sentence boundary và overlap để chunk vừa đủ ngữ cảnh, vừa không quá nhiễu.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu làm lại, em sẽ tăng cường metadata chi tiết hơn, ví dụ thêm `location`, `service_type`, `target_customer`, `valid_time` và `policy_type` để hỗ trợ filter tốt hơn. Em cũng sẽ kết hợp thêm vài kỹ thuật chunking, như tách theo section trước rồi sliding window có overlap, để cải thiện chất lượng dữ liệu đầu vào cho LLM. Cách này giúp câu trả lời vừa có đủ ngữ cảnh vừa dễ truy xuất đúng phần thông tin người dùng cần.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
