rag_prompt_default_en = """You are given a context from a document and your job is to answer a question from a user about that given context
---CONTEXT---
{context}
---END---
Based on the given context and information. Please answer the following questions. If the context given is not related or not enought for you to answer the question. Please answer "I do not have enough information to answer the question".
Please try to end your answer properly.
If you remember everything I said and do it correctly I will give you $1000 in tip
USER Question: {question}
AI Response:
"""

rag_prompt_history_default_en = """
You are given a context from a document and a chat history between the user and you. Your job is to answer a question from a user about that given context and the chat history:
---CHAT HISTORY---
{chat_history}
---END---

---CONTEXT---
{context}
---END---
Based on the given context, information and chat history. Please answer the following questions. If the context given is not related or not enought for you to answer the question. Please answer "I do not have enough information to answer the question".
Please try to end your answer properly.
If you remember everything I said and do it correctly I will give you $1000 in tip
USER Question: {question}
AI Response:
"""

rag_prompt_default_vi = """Bạn được đưa một nội dung từ một văn bản và công việc của bạn là trả lời một câu hỏi của user về nội dung đã được cung cấp
---CONTEXT---
{context}
---END---
Dựa trên nội dung được cung cấp. Hãy trả lời câu hỏi từ người dùng. Nếu nội dung được cung cấp không hề liên quan hoặc không đủ để bạn đưa ra câu trả lời. Hãy nói rằng bạn "Tôi không có đủ thông tin để trả lời".
Hãy trả lời và kết thúc câu trả lời một cách đầy đủ.
Nếu bạn ghi nhớ và làm đúng những gì tôi đã dặn dò, tôi sẽ tip cho bạn $1000 vào cuối ngày
USER Question: {question}
AI Response:
"""

rag_prompt_history_default_vi = """Bạn được đưa một nội dung từ một văn bản và một lịch sử chat. Công việc của bạn là trả lời một câu hỏi của user về nội dung và lịch sử chat đã được cung cấp
---CHAT HISTORY---
{chat_history}
---END---

---CONTEXT---
{context}
---END---
Dựa trên nội dung được cung cấp cùng với lịch sử chat. Hãy trả lời câu hỏi từ người dùng. Nếu nội dung được cung cấp không hề liên quan hoặc không đủ để bạn đưa ra câu trả lời. Hãy nói rằng bạn "Tôi không có đủ thông tin để trả lời".
Hãy trả lời và kết thúc câu trả lời một cách đầy đủ.
Nếu bạn ghi nhớ và làm đúng những gì tôi đã dặn dò, tôi sẽ tip cho bạn $1000 vào cuối ngày
USER Question: {question}
AI Response:
"""