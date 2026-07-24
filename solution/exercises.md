# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng placeholder trả lời bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature thấp như 0.0, câu trả lời thường ổn định, trực tiếp và ít biến đổi giữa các lần gọi. Khi tăng lên 1.0 hoặc 1.5, phản hồi có xu hướng sáng tạo hơn, dùng cách diễn đạt đa dạng hơn, nhưng cũng dễ lan man hoặc chọn chi tiết ít chắc chắn hơn. Vì vậy temperature càng cao thì tính bất ngờ tăng, còn độ nhất quán giảm.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Với chatbot hỗ trợ khách hàng, tôi sẽ chọn temperature khoảng 0.2 đến 0.4. Trường hợp này cần câu trả lời nhất quán, đúng chính sách và ít bịa hơn là quá sáng tạo. Nếu cần giọng văn thân thiện hơn, có thể tăng nhẹ nhưng vẫn nên giữ dưới 0.5.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Theo bảng giá output trong file, GPT-4o là 0.010 USD/1K token còn GPT-4o-mini là 0.0006 USD/1K token, tức GPT-4o đắt hơn khoảng 16.7 lần cho phần output. Workload này có 10.000 * 3 * 350 = 10.500.000 output token mỗi ngày, nên GPT-4o tốn khoảng 105 USD/ngày còn mini khoảng 6.3 USD/ngày. GPT-4o xứng đáng khi cần lập luận khó, độ chính xác cao hoặc xử lý tình huống quan trọng; mini phù hợp cho FAQ, phân loại, tóm tắt ngắn hoặc chatbot lưu lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona giáo viên tiểu học, câu trả lời thường ngắn hơn, dùng từ đơn giản và ví dụ gần gũi như cuốn sổ chia sẻ giữa nhiều người. Với persona chuyên gia tài chính, phản hồi dài và kỹ thuật hơn, có thể nhắc đến sổ cái phân tán, đồng thuận, giao dịch, rủi ro và ứng dụng tài chính. System prompt ảnh hưởng mạnh đến vai trò, mức độ chuyên sâu, cách chọn ví dụ và giọng văn của model. Cùng một câu hỏi user nhưng model sẽ ưu tiên cách giải thích phù hợp với persona được đặt trước.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Khi so sánh một đoạn tiếng Việt khoảng 100 từ, số token từ tiktoken thường cao hơn ước lượng số từ / 0.75, có thể lệch khoảng 20% đến 50% tùy đoạn văn và dấu tiếng Việt. Ước lượng theo số từ chỉ là cách tính thô, còn tokenizer chia văn bản theo mảnh token thực tế. Tiếng Việt thường tốn nhiều token hơn tiếng Anh vì có dấu, nhiều từ ghép được viết tách bằng khoảng trắng, và tokenizer thường không gộp các cụm tiếng Việt hiệu quả bằng các từ tiếng Anh phổ biến.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi phản hồi dài hoặc người dùng cần cảm giác hệ thống đang xử lý ngay, ví dụ chatbot tư vấn, viết nội dung dài, giải thích từng bước hoặc trợ lý CLI tương tác. Người dùng không phải chờ toàn bộ câu trả lời xong mới thấy kết quả, nên trải nghiệm mượt hơn. Non-streaming phù hợp hơn khi cần xử lý kết quả như một khối hoàn chỉnh, ví dụ gọi API backend để lấy JSON, chấm điểm, phân loại, logging, hoặc các tác vụ mà giao diện chỉ cần hiện kết quả cuối cùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giảm áp lực lên API khi hệ thống đang quá tải bằng cách giãn thời gian retry ngày càng dài hơn sau mỗi lần lỗi. So với delay cố định, nó cho server thêm thời gian hồi phục và giảm số request bị dồn vào cùng một thời điểm. Nếu hàng nghìn client đều retry sau đúng 1 giây, chúng có thể tạo thành các đợt request đồng loạt, làm tình trạng quá tải lặp lại và khiến lỗi kéo dài hơn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona tôi chọn là: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt, giải thích từng bước khi người học gặp lỗi code hoặc chưa hiểu khái niệm." Tôi dùng "trợ giảng thân thiện" để model giữ giọng hỗ trợ, kiên nhẫn và phù hợp môi trường học tập. Tôi yêu cầu "trả lời ngắn gọn bằng tiếng Việt" vì người học cần câu trả lời dễ đọc, đúng ngôn ngữ, không bị quá tải bởi giải thích dài không cần thiết.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là trợ lý chỉ giữ history 3 lượt gần nhất, nên dễ quên bối cảnh dài hoặc quyết định đã thống nhất trước đó. Một cải thiện cụ thể là thêm bộ nhớ tóm tắt: sau mỗi vài lượt, tạo một bản summary ngắn của các thông tin quan trọng rồi đưa summary này vào system/context ở các lượt sau. Cách triển khai là lưu `conversation_summary`, cập nhật nó bằng một lời gọi model riêng hoặc hàm tóm tắt định kỳ, sau đó gửi summary cùng với history ngắn khi gọi API.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
