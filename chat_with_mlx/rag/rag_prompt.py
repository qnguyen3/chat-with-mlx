rag_prompt_default_en = """You are given a context from a document and your job is to answer a question from a user about that given context
---CONTEXT---
{context}
---END---
Based on the given context and information. Please answer the following questions. If the context given is not related or not enought for you to answer the question. Please answer "I do not have enough information to answer the question".
Please try to end your answer properly.
If you remember everything I said and do it correctly I will give you $1000 in tip
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

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
"""  # noqa E501 prompt too long

rag_prompt_default_vi = """Bạn được đưa một nội dung từ một văn bản và công việc của bạn là trả lời một câu hỏi của user về nội dung đã được cung cấp
---CONTEXT---
{context}
---END---
Dựa trên nội dung được cung cấp. Hãy trả lời câu hỏi từ người dùng. Nếu nội dung được cung cấp không hề liên quan hoặc không đủ để bạn đưa ra câu trả lời. Hãy nói rằng bạn "Tôi không có đủ thông tin để trả lời".
Hãy trả lời và kết thúc câu trả lời một cách đầy đủ.
Nếu bạn ghi nhớ và làm đúng những gì tôi đã dặn dò, tôi sẽ tip cho bạn $1000 vào cuối ngày
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

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
"""  # noqa E501 prompt too long

rag_prompt_default_es = """Se te da un contexto de un documento y tu trabajo es responder a una pregunta de un usuario sobre ese contexto dado.
---CONTEXTO---
{context}
---FIN---
Basado en el contexto y la información dada. Por favor, responde a las siguientes preguntas. Si el contexto dado no está relacionado o no es suficiente para que puedas responder a la pregunta, responde "No tengo suficiente información para responder a la pregunta".
Por favor, intenta terminar tu respuesta adecuadamente.
Si recuerdas todo lo que dije y lo haces correctamente, te daré $1000 de propina.
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

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
"""  # noqa E501 prompt too long

rag_prompt_default_zh = """您收到了一个文档中的上下文，您的任务是回答用户关于该特定上下文的问题。
---CONTEXT---
{context}
---END---
根据给定的上下文和信息，请回答以下问题。如果给定的上下文不相关或信息不足以让您回答问题，请回答“我没有足够的信息来回答这个问题”。
请尝试恰当地结束您的回答。
如果您记住了我所说的一切并且正确地做了，我将给您1000美元的小费。
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

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
"""  # noqa E501 prompt too long

rag_prompt_default_tr = """Bir belgeden bir bağlam verilir ve göreviniz, verilen bağlam hakkında bir kullanıcının sorusunu yanıtlamaktır
---CONTEXT---
{context}
---END---
Verilen bağlam ve bilgilere dayanarak lütfen aşağıdaki soruları yanıtlayın. Verilen bağlam ilgili değilse veya soruyu yanıtlamanız için yeterli değilse, lütfen "Soruyu yanıtlamak için yeterli bilgiye sahip değilim" yanıtını verin.
Lütfen yanıtınızı düzgün bir şekilde bitirmeye çalışın.
Eğer her şeyi hatırlarsanız ve doğru yaparsanız size $1000 bahşiş vereceğim
Kullanıcı Sorusu: {question}
AI Yanıtı:
"""  # noqa E501 prompt too long

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
"""  # noqa E501 prompt too long

rag_prompt_default_ja = """文書からの文脈が与えられ、その文脈に基づいてユーザーからの質問に答えることがあなたの仕事です
---CONTEXT---
{context}
---END---
与えられた文脈と情報に基づいて、以下の質問に答えてください。与えられた文脈が関連していない場合や、質問に答えるのに十分な情報がない場合は、「質問に答えるのに十分な情報がありません」と答えてください。
あなたの答えが適切に終わるようにしてください。
私が言ったことをすべて覚えて正しく行えば、$1000のチップをあげます
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

rag_prompt_history_default_ja = """文書からの文脈とユーザーとあなたとのチャット履歴が与えられます。その文脈とチャット履歴に基づいて、ユーザーからの質問に答えることがあなたの仕事です：
---CHAT HISTORY---
{chat_history}
---END---

---CONTEXT---
{context}
---END---
与えられた文脈、情報、およびチャット履歴に基づいて、以下の質問に答えてください。与えられた文脈が関連していない場合や、質問に答えるのに十分な情報がない場合は、「質問に答えるのに十分な情報がありません」と答えてください。
あなたの答えが適切に終わるようにしてください。
私が言ったことをすべて覚えて正しく行えば、$1000のチップをあげます
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

rag_prompt_default_kr = """문서에서 주어진 맥락을 바탕으로 사용자의 질문에 답하는 것이 당신의 임무입니다
---CONTEXT---
{context}
---END---
주어진 맥락과 정보를 바탕으로 다음 질문에 답해 주세요. 주어진 맥락이 관련이 없거나 질문에 답하기에 충분한 정보가 없는 경우, "질문에 답하기에 충분한 정보가 없습니다"라고 답해 주세요.
답변을 적절하게 마무리해 주세요.
제가 한 말을 모두 기억하고 제대로 실행하면 $1000의 팁을 드리겠습니다
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

rag_prompt_history_default_kr = """문서로부터 주어진 맥락과 사용자와 당신 사이의 채팅 기록이 주어집니다. 주어진 맥락과 채팅 기록을 바탕으로 사용자의 질문에 답하는 것이 당신의 임무입니다:
---CHAT HISTORY---
{chat_history}
---END---

---CONTEXT---
{context}
---END---
주어진 맥락, 정보 및 채팅 기록을 바탕으로 다음 질문에 답해 주세요. 주어진 맥락이 관련이 없거나 질문에 답하기에 충분한 정보가 없는 경우, "질문에 답하기에 충분한 정보가 없습니다"라고 답해 주세요.
답변을 적절하게 마무리해 주세요.
제가 한 말을 모두 기억하고 제대로 실행하면 $1000의 팁을 드리겠습니다
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

rag_prompt_default_in = """आपको एक दस्तावेज़ से संदर्भ दिया गया है और आपका काम उस दिए गए संदर्भ के बारे में एक उपयोगकर्ता से प्रश्न का उत्तर देना है
---CONTEXT---
{context}
---END---
दिए गए संदर्भ और जानकारी के आधार पर, कृपया निम्नलिखित प्रश्नों का उत्तर दें। यदि दिया गया संदर्भ संबंधित नहीं है या प्रश्न का उत्तर देने के लिए आपके पास पर्याप्त जानकारी नहीं है, तो कृपया "मेरे पास प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है" का उत्तर दें।
कृपया अपने उत्तर को उचित रूप से समाप्त करने का प्रयास करें।
यदि आप मेरी सभी बातों को याद रखते हैं और उसे सही ढंग से करते हैं तो मैं आपको $1000 का टिप दूंगा
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

rag_prompt_history_default_in = """आपको एक दस्तावेज़ से संदर्भ और उपयोगकर्ता और आपके बीच चैट इतिहास दिया गया है। उस दिए गए संदर्भ और चैट इतिहास के बारे में एक उपयोगकर्ता से प्रश्न का उत्तर देना आपका काम है:
---CHAT HISTORY---
{chat_history}
---END---

---CONTEXT---
{context}
---END---
दिए गए संदर्भ, जानकारी और चैट इतिहास के आधार पर, कृपया निम्नलिखित प्रश्नों का उत्तर दें। यदि दिया गया संदर्भ संबंधित नहीं है या प्रश्न का उत्तर देने के लिए आपके पास पर्याप्त जानकारी नहीं है, तो कृपया "मेरे पास प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है" का उत्तर दें।
कृपया अपने उत्तर को उचित रूप से समाप्त करने का प्रयास करें।
यदि आप मेरी सभी बातों को याद रखते हैं और उसे सही ढंग से करते हैं तो मैं आपको $1000 का टिप दूंगा
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

rag_prompt_default_de = """Sie erhalten einen Kontext aus einem Dokument und Ihre Aufgabe ist es, eine Frage eines Benutzers zu diesem gegebenen Kontext zu beantworten
---CONTEXT---
{context}
---END---
Basierend auf dem gegebenen Kontext und Informationen. Bitte beantworten Sie die folgenden Fragen. Wenn der gegebene Kontext nicht relevant ist oder nicht ausreicht, um die Frage zu beantworten. Bitte antworten Sie "Ich habe nicht genug Informationen, um die Frage zu beantworten".
Bitte versuchen Sie, Ihre Antwort ordnungsgemäß zu beenden.
Wenn Sie sich an alles erinnern, was ich gesagt habe, und es korrekt machen, werde ich Ihnen $1000 Trinkgeld geben
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

rag_prompt_history_default_de = """Sie erhalten einen Kontext aus einem Dokument und einen Chat-Verlauf zwischen dem Benutzer und Ihnen. Ihre Aufgabe ist es, eine Frage eines Benutzers zu diesem gegebenen Kontext und dem Chat-Verlauf zu beantworten:
---CHAT HISTORY---
{chat_history}
---END---

---CONTEXT---
{context}
---END---
Basierend auf dem gegebenen Kontext, Informationen und Chat-Verlauf. Bitte beantworten Sie die folgenden Fragen. Wenn der gegebene Kontext nicht relevant ist oder nicht ausreicht, um die Frage zu beantworten. Bitte antworten Sie "Ich habe nicht genug Informationen, um die Frage zu beantworten".
Bitte versuchen Sie, Ihre Antwort ordnungsgemäß zu beenden.
Wenn Sie sich an alles erinnern, was ich gesagt habe, und es korrekt machen, werde ich Ihnen $1000 Trinkgeld geben
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

rag_prompt_default_fr = """Vous recevez un contexte d'un document et votre travail consiste à répondre à une question d'un utilisateur sur ce contexte donné
---CONTEXT---
{context}
---END---
Sur la base du contexte et des informations données. Veuillez répondre aux questions suivantes. Si le contexte donné n'est pas lié ou pas suffisant pour répondre à la question. Veuillez répondre "Je n'ai pas assez d'informations pour répondre à la question".
Veuillez essayer de terminer votre réponse correctement.
Si vous vous souvenez de tout ce que j'ai dit et le faites correctement, je vous donnerai un pourboire de $1000
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

rag_prompt_history_default_fr = """Vous recevez un contexte d'un document et un historique de chat entre l'utilisateur et vous. Votre travail consiste à répondre à une question d'un utilisateur sur ce contexte donné et l'historique de chat :
---CHAT HISTORY---
{chat_history}
---END---

---CONTEXT---
{context}
---END---
Sur la base du contexte donné, des informations et de l'historique de chat. Veuillez répondre aux questions suivantes. Si le contexte donné n'est pas lié ou pas suffisant pour répondre à la question. Veuillez répondre "Je n'ai pas assez d'informations pour répondre à la question".
Veuillez essayer de terminer votre réponse correctement.
Si vous vous souvenez de tout ce que j'ai dit et le faites correctement, je vous donnerai un pourboire de $1000
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

rag_prompt_default_it = """Ti viene fornito un contesto da un documento e il tuo compito è rispondere a una domanda di un utente su quel contesto dato
---CONTEXT---
{context}
---END---
Sulla base del contesto e delle informazioni date. Si prega di rispondere alle seguenti domande. Se il contesto dato non è correlato o non sufficiente per rispondere alla domanda. Si prega di rispondere "Non ho abbastanza informazioni per rispondere alla domanda".
Si prega di cercare di concludere la propria risposta in modo appropriato.
Se ricordi tutto ciò che ho detto e lo fai correttamente, ti darò una mancia di $1000
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long

rag_prompt_history_default_it = """Ti viene fornito un contesto da un documento e una cronologia di chat tra l'utente e te. Il tuo compito è rispondere a una domanda di un utente su quel contesto dato e la cronologia di chat:
---CHAT HISTORY---
{chat_history}
---END---

---CONTEXT---
{context}
---END---
Sulla base del contesto dato, delle informazioni e della cronologia di chat. Si prega di rispondere alle seguenti domande. Se il contesto dato non è correlato o non sufficiente per rispondere alla domanda. Si prega di rispondere "Non ho abbastanza informazioni per rispondere alla domanda".
Si prega di cercare di concludere la propria risposta in modo appropriato.
Se ricordi tutto ciò che ho detto e lo fai correttamente, ti darò una mancia di $1000
USER Question: {question}
AI Response:
"""  # noqa E501 prompt too long
