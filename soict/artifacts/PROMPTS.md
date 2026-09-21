# Prompts

The IterCOMP paper describes its three prompted steps but does not release the
prompt text, so these were written from the method description in Section 3 of
that paper. They are reproduced here in full because the paper that uses them
cites this file rather than printing them.

**What actually runs is the Vietnamese version.** The English text below is a
translation, given so that the prompts can be read without Vietnamese. The two
are kept line-for-line parallel; if you edit one, edit the other.

Placeholders in `{...}` are substituted at run time.

---

## 1. Answerability judgment (Step 2)

Decides whether the evidence gathered so far is enough to stop the loop.
Source: [`src/itercomp/core.py`](src/itercomp/core.py) — `ANSWERABLE_PROMPT`.

**Vietnamese (as run):**

```
Bạn đang giải một câu hỏi multi-hop. Dưới đây là các bằng chứng đã thu thập.

CÂU HỎI: {question}

BẰNG CHỨNG ĐÃ CÓ:
{evidence}

Các bằng chứng trên đã ĐỦ để trả lời chính xác câu hỏi chưa?
Chỉ trả lời một từ: ANSWERABLE_YES hoặc ANSWERABLE_NO
```

**English:**

```
You are solving a multi-hop question.
Below is the evidence collected so far.

QUESTION: {question}

EVIDENCE SO FAR:
{evidence}

Is the evidence above SUFFICIENT to answer the question accurately?
Reply with exactly one word: ANSWERABLE_YES or ANSWERABLE_NO
```

The reply is read as a single token. Anything that is not `ANSWERABLE_YES`
counts as "not yet", so a malformed reply continues the loop rather than
ending it early.

---

## 2. Follow-up question generation (Step 3)

Names the missing information, and that follow-up becomes the query for the
next retrieval pass. Source: [`src/itercomp/core.py`](src/itercomp/core.py) —
`FOLLOWUP_PROMPT`.

**Vietnamese (as run):**

```
Bạn đang giải một câu hỏi multi-hop nhưng bằng chứng chưa đủ.

CÂU HỎI GỐC: {question}

BẰNG CHỨNG ĐÃ CÓ:
{evidence}

Hãy nêu MỘT câu hỏi phụ ngắn gọn, nhắm vào đúng thông tin còn THIẾU
để có thể trả lời câu hỏi gốc. Chỉ ghi câu hỏi phụ, không giải thích.

FOLLOW-UP:
```

**English:**

```
You are solving a multi-hop question but the evidence is not yet sufficient.

ORIGINAL QUESTION: {question}

EVIDENCE SO FAR:
{evidence}

State ONE short follow-up question targeting exactly the MISSING information
needed to answer the original question. Give only the follow-up question, no
explanation.

FOLLOW-UP:
```

---

## 3. Reader prompt

Shared by **every** compression method evaluated, so that comparisons differ
only in the context supplied and never in how the answer is requested. This one
is bilingual in the code itself, because the same reader is used on English and
Vietnamese datasets. Source:
[`src/itercomp/reader.py`](src/itercomp/reader.py) — `READER_PROMPT`.

```
Answer the question using ONLY the context below.
Reply with the SHORT answer only (a few words, no explanation, no full sentence).
If the context does not contain the answer, reply: unknown

Trả lời câu hỏi CHỈ dựa vào ngữ cảnh dưới đây.
Chỉ ghi đáp án NGẮN GỌN (vài từ, không giải thích, không viết thành câu).

CONTEXT:
{context}

QUESTION: {question}
ANSWER:
```

Asking for a short answer matters for the metric: F1 is computed over tokens,
so a model that answers in a full sentence is penalised for words that carry no
information. The `unknown` escape keeps a guess from scoring by accident.
