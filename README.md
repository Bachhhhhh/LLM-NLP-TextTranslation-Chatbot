# Project: LLM-NLP-TextTranslation-Chatbot

---

## Table of Contents

1. [Overview](#overview)
2. [Library Requirements](#requirements)
3. [Question 1 — Single & Multiple Text Translation](#question-1)
4. [Question 2 — Chatbot Development](#question-2)
5. [Chatbot Test Set & Evaluation](#chatbot-test-set-&-evaluation)
6. [Conclusion](#conclusion)
7. [Contributors](#contributors)

---

## Overview

This project demonstrates how to integrate a large language model (Gemini) for two main tasks:

1. **Single text translation**: translate a single text into a destination language, but return the same text if it is already in the destination language.
2. **Multiple texts translation**: translate a list of texts into a destination language, preserving any texts that are already in that language.

Additionally, the project implements a **retrieval-based chatbot** that:

* Crawls and extracts text from the provided URL.
* Cleans and preprocesses the text for embeddings.
* Builds an embedding index and performs nearest-neighbor retrieval for queries.
* Uses Gemini to generate a final answer from retrieved context.

---

## Library Requirements

* python 3.8+
* torch 2.0.0+
* sentence-transformers 2.2.2+
* numpy 2.0.0+
* qdm 4.65.0+
* transformers 4.30.0+
* scikit-learn 1.2.2+

---

## Question 1 — Single & Multiple Text Translation

### Goal

* Implement a function to translate a **single** text to a destination language using the Gemini API, but first **check whether the text is already in the destination language** and return it unchanged if so.
* Implement a function to translate a **list** of texts and return their translations preserving texts that are already the destination language.

### JSON Inputs

```python
json_1 = {
    'text': 'Hello',
    'dest_language': 'vi'
}

json_2 = {
    'text': ['Hello', 'I am John','Tôi là sinh viên'],
    'dest_language': 'vi'
}
```

### Expected Output
```
Output 1: 'Xin chào'

Output 2: ['Xin chào', 'Tôi là John', 'Tôi là sinh viên']
```

### Implementation notes

* The repository includes two functions (Python) for this task:

  * `translate_single_text(json_input)` — accepts `json_1` style input, checks language, and returns the translation.
  * `translate_multiple_texts(json_input)` — accepts `json_2` style input, loops over the list and returns a list of translations.

---

## Question 2 — Chatbot Development

### Goal

Build a chatbot that answers questions based on text extracted from a website. The target dataset is the text content displayed on the webpage 'https://www.presight.io/privacy-policy.html', extracted directly from the website's HTML structure.

The chatbot should:

1. Crawl and extract text from the website.
2. Clean and preprocess the text for embeddings.
3. Build an embedding index (vector store) for retrieval.
4. Given a user query, find the most relevant paragraph(s) and then use Gemini to generate a precise answer using that context.

#### Note

This page is no longer available, the URL now returns no content. This project uses previously crawled data for demonstration purposes.

### Implemented methods

The following functions are implemented in the Chatbot class includes:

* `def __init__(self, api_key, url):` — Initialize the chatbot configuration (API key, target URL, models, local index path, etc).

* `def crawl_data(self):` — Crawl the target website and extract text using BeautifulSoup. The method collects textual blocks (e.g., paragraphs, headings) and stores them in a local data structure for further processing.

* `def clean(self, text):` — Clean raw text before embedding: lowercasing, removing special characters, normalizing whitespace, and other normalization steps.

* `def search_query(self, query, top_k):` — Search for relevant text chunks given a `query`:

* `def generate_answer_with_gemini(self, context, question):` — Format a prompt that includes the retrieved `context` and the `question`, and call Gemini (e.g., Gemini Flash 1.5) to generate an accurate answer. 

* `def make_request(self, question):` — cleans the question, retrieves top‑k similar text chunks using search_query(), measures retrieval time, and finally calls generate_answer_with_gemini() to produce the final LLM‑based answer.


### Retrieval Pipeline

The Code Flow could be summarized in 3 main phases:

1. Crawl -> chunk -> clean -> embed -> store embeddings (and original chunks).
2. Query -> embed query -> nearest neighbors (cosine similarity) -> select top-k chunks.
3. Construct a prompt with selected chunks as context -> call Gemini for final answer.

---

## Chatbot Test Set & Evaluation

The project contains a test harness that runs the chatbot against a curated set of questions. The question set are stored in a Python list of dictionaries below:

```python
questions = [
    {"section": "URELATED QUESTION",
     "question": "What is policy?"},

    {"section": "PRIVACY POLICY - Last updated 15 Sep 2023",
     "question": "When was the privacy policy last updated?"},
    
    {"section": "Commitment to Privacy",
     "question": "What does Presight commit to protecting for its customers and visitors?"},
    
    {"section": "Privacy Policy Explanation",
     "question": "What does this Privacy Policy explain regarding customers and visitors?"},
    
    {"section": "INFORMATION COLLECTION AND USE",
     "question": "Why does Presight collect different types of information?"},
    
    {"section": "TYPES OF DATA COLLECTED - PERSONAL DATA",
     "question": "What types of personally identifiable information does Presight collect?"},
    
    {"section": "TYPES OF DATA COLLECTED - USAGE DATA",
     "question": "What is included in the usage data collected by Presight?"},
    
    {"section": "USE OF DATA",
     "question": "For what purposes does Presight use the collected data?"},
    
    {"section": "CONSENT",
     "question": "How does Presight ensure that personal information submitted is correct?"},
    
    {"section": "ACCESS TO PERSONAL INFORMATION - ACCESSING YOUR PERSONAL INFORMATION",
     "question": "How can users access and update their personal information held by Presight?"},
    
    {"section": "ACCESS TO PERSONAL INFORMATION - AUTOMATED EDIT CHECKS",
     "question": "What is the purpose of automated edit checks in collecting personal information?"},
    
    {"section": "DISCLOSURE OF INFORMATION",
     "question": "Under what circumstances might Presight disclose application data to third parties?"},
    
    {"section": "SHARING OF PERSONAL DATA",
     "question": "Does Presight share personal data with third parties or AI models?"},
    
    {"section": "GOOGLE USER DATA AND GOOGLE WORKSPACE APIS",
     "question": "What restrictions does Presight place on the use of Google User Data and Google Workspace APIs?"},
    
    {"section": "DATA SECURITY",
     "question": "What encryption and security measures does Presight use to protect customer data?"},
    
    {"section": "DATA RETENTION & DISPOSAL",
     "question": "How long does Presight retain customer data after account closure?"},
    
    {"section": "QUALITY, INCLUDING DATA SUBJECTS' RESPONSIBILITIES FOR QUALITY",
     "question": "What responsibilities do users have in maintaining the accuracy of their personal data?"},
    
    {"section": "MONITORING AND ENFORCEMENT",
     "question": "What actions does Presight take to monitor data compliance and handle data breaches?"},
    
    {"section": "COOKIES",
     "question": "How can users manage cookies on Presight’s website?"},
    
    {"section": "THIRD-PARTY WEBSITES",
     "question": "What is Presight’s responsibility regarding third-party websites?"},
    
    {"section": "CHANGES TO PRIVACY POLICY",
     "question": "Where will updates to the Privacy Policy be posted?"},
    
    {"section": "CONTACT US",
     "question": "What is the email address provided for contacting Presight if I have questions about the Privacy Policy?"},
    
    {"section": "PURPOSEFUL USE ONLY",
     "question": "For what purposes does Presight commit to using personal information?"}
]
```

### Test harness loop

The included harness runs the chatbot over the question set with a small random delay between requests for debugging and evaluation. Example code snippet:

```python
# === Auto query loop ===
for item in questions:
    delay = random.uniform(0, 2)
    time.sleep(delay)
    print("\n===== Debug Information =====")
    print(f"Section: {item['section']}")  
    print(f"Question: {item['question']}")
    chatbot.make_request(item['question'])
```

### Output Example

Below is an sample output demonstrating how the chatbot handles retrieval, context scoring, and final answers.

```
===== Debug Information =====
Section: PRIVACY POLICY - Last updated 15 Sep 2023
Question: When was the privacy policy last updated?
Context :
- (Score: 0.8726) PRIVACY POLICY - Last updated 15 Sep 2023
- (Score: 0.7766) CHANGES TO PRIVACY POLICY: We may update this Privacy Policy from time to time. The updated Privacy Policy will be posted on our website.
- (Score: 0.5045) This Privacy Policy explains how we collect, use, and disclose information about our customers and visitors.
Chatbot : The privacy policy was last updated on September 15, 2023.
Running time: 0.62 seconds
```

### Evaluation

* In general, the Chatbot answers correctly and contextually, it consistently selects the most relevant website sections.

* Similarity scoring really works well, which proved by the corresponding between High-scoring context items and the accurate answers.

* All the outputs are generated in under 1 second, this makes the Chatbot suitable for real-time use.

---

## Conclusion

Through this project, my teammates and I learned how to build a complete retrieval-augmented chatbot pipeline—from extracting and cleaning website text, to embedding data, performing similarity search, and finally generating answers using Gemini. 
Working with real website content helped us understand how to handle unstructured text and prepare it for retrieval. 
Implementing the RAG workflow (instead of using a prebuilt framework) deepened our understanding of how context retrieval influences answer quality, while integrating Gemini demonstrated how an LLM can generate accurate, context-aware responses. 
Additionally, the debugging process—logging similarity scores, retrieved context, and runtime—helped reinforce best practices for evaluating and validating system performance. 
Overall, this project provided a solid foundation in combining retrieval methods with LLMs to build a lightweight, effective domain-specific chatbot.

---

## Contributors

* Pham Ba Hoang Anh
* Truong Binh Ba
* Pham Le Hong Duc
* Bach Ngoc Le Duy
