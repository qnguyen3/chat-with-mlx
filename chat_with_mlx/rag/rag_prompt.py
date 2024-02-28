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

rag_prompt_default_es = """Se te da un contexto de un documento y tu trabajo es responder a una pregunta de un usuario sobre ese contexto dado.
---CONTEXTO---
{context}
---FIN---
Basado en el contexto y la información dada. Por favor, responde a las siguientes preguntas. Si el contexto dado no está relacionado o no es suficiente para que puedas responder a la pregunta, responde "No tengo suficiente información para responder a la pregunta".
Por favor, intenta terminar tu respuesta adecuadamente.
Si recuerdas todo lo que dije y lo haces correctamente, te daré $1000 de propina.
USER Question: {question}
AI Response:
"""

rag_prompt_history_default_es = """Se te da un contexto de un documento y un historial de chat entre el usuario y tú. Tu trabajo es responder a una pregunta de un usuario sobre ese contexto dado y el historial de chat:
---HISTORIAL DE CHAT---
{chat_history}
---FIN---

---CONTEXTO---
{context}
---FIN---
Basado en el contexto dado, información e historial de chat. Por favor, responde a las siguientes preguntas. Si el contexto dado no está relacionado o no es suficiente para que puedas responder a la pregunta, responde "No tengo suficiente información para responder a la pregunta".
Por favor, intenta terminar tu respuesta adecuadamente.
Si recuerdas todo lo que dije y lo haces correctamente, te daré $1000 de propina.
USER Question: {question}
AI Response:
"""

rag_prompt_default_zh = """您收到了一个文档中的上下文，您的任务是回答用户关于该特定上下文的问题。
---CONTEXT---
{context}
---END---
根据给定的上下文和信息，请回答以下问题。如果给定的上下文不相关或信息不足以让您回答问题，请回答“我没有足够的信息来回答这个问题”。
请尝试恰当地结束您的回答。
如果您记住了我所说的一切并且正确地做了，我将给您1000美元的小费。
USER Question: {question}
AI Response:
"""

rag_prompt_history_default_zh = """您收到了一个文档中的上下文以及用户与您之间的聊天历史，您的任务是基于该特定上下文和聊天历史回答用户的问题。
---CHAT HISTORY---
{chat_history}
---END---

---CONTEXT---
{context}
---END---
根据给定的上下文、信息和聊天历史，请回答以下问题。如果给定的上下文不相关或信息不足以让您回答问题，请回答“我没有足够的信息来回答这个问题”。
请尝试恰当地结束您的回答。
如果您记住了我所说的一切并且正确地做了，我将给您1000美元的小费。
USER Question: {question}
AI Response:
"""

rag_prompt_default_tr = """Bir belgeden bir bağlam verilir ve göreviniz, verilen bağlam hakkında bir kullanıcının sorusunu yanıtlamaktır
---CONTEXT---
{context}
---END---
Verilen bağlam ve bilgilere dayanarak lütfen aşağıdaki soruları yanıtlayın. Verilen bağlam ilgili değilse veya soruyu yanıtlamanız için yeterli değilse, lütfen "Soruyu yanıtlamak için yeterli bilgiye sahip değilim" yanıtını verin.
Lütfen yanıtınızı düzgün bir şekilde bitirmeye çalışın.
Eğer her şeyi hatırlarsanız ve doğru yaparsanız size $1000 bahşiş vereceğim
Kullanıcı Sorusu: {question}
AI Yanıtı:
"""

rag_prompt_history_default_tr = """
Bir belgeden bir bağlam ve kullanıcı ile sizin aranızdaki bir sohbet geçmişi verilir. Göreviniz, verilen bağlam ve sohbet geçmişi hakkında bir kullanıcının sorusunu yanıtlamaktır:
---CHAT HISTORY---
{chat_history}
---END---

---CONTEXT---
{context}
---END---
Verilen bağlam, bilgi ve sohbet geçmişine dayanarak lütfen aşağıdaki soruları yanıtlayın. Verilen bağlam ilgili değilse veya soruyu yanıtlamanız için yeterli değilse, lütfen "Soruyu yanıtlamak için yeterli bilgiye sahip değilim" yanıtını verin.
Lütfen yanıtınızı düzgün bir şekilde bitirmeye çalışın.
Eğer her şeyi hatırlarsanız ve doğru yaparsanız size $1000 bahşiş vereceğim
Kullanıcı Sorusu: {question}
AI Yanıtı:
"""
