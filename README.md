# Medical Chatbot for Cancer Information Retrieval

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Business Understanding & Stakeholders](#2-business-understanding--stakeholders)
3.  [Dataset Overview](#3-dataset-overview)
4.  [Data Preprocessing](#4-data-preprocessing)
5.  [Modelling: Text Representation](#5-modelling-text-representation)
6.  [Hybrid Search System](#6-hybrid-search-system)
7.  [Model Evaluation](#7-model-evaluation)
8.  [Optimization Results & Final Accuracy](#8-optimization-results--final-accuracy)
9.  [Challenges Encountered](#9-challenges-encountered)
10. [User Interface](#10-user-interface)
11. [Future Improvements & Recommendations](#11-future-improvements--recommendations)
12. [How to Run the Project Locally](#12-how-to-run-the-project-locally)
13. [Contact / Contributors](#13-contact--contributors)

---

## 1. Introduction

This project details the development of a robust medical chatbot designed to provide accurate and timely cancer-related information. Our primary goal is to address the prevalent issue of misinformation and the strain on healthcare professionals by offering an accessible, reliable, and intelligent information retrieval system.

## 2. Business Understanding & Stakeholders

**Problem Statement:** Patients and the general public often struggle to find trustworthy, quick answers to their cancer-related questions online, leading to anxiety and increasing the burden on healthcare providers.

**Project Objectives:**
* Develop a robust text preprocessing pipeline for complex medical data.
* Implement an effective hybrid information retrieval mechanism.
* Rigorously evaluate model performance to ensure accuracy and relevance.
* Prepare the core NLP model for seamless API integration.
* Enhance the user experience by delivering fast, reliable, and relevant information.

**Key Stakeholders:**
* **Patients/General Public:** Direct beneficiaries seeking validated health information.
* **Healthcare Providers/Hospitals:** Can leverage the chatbot to streamline patient support and reduce routine inquiries.
* **Medical Researchers/Educators:** Can utilize the structured data for analysis and educational content development.

## 3. Dataset Overview

Our project is built upon the **MedQuAD (Medical Question-Answer Dataset)**.
* **Source:** This dataset comprises curated Q&A pairs extracted from trusted medical websites like Cancer.gov, ensuring high data credibility.
* **Format:** The data is provided in XML files, which we processed using a custom loader.
* **Initial Size:** The raw dataset contained 729 Question-Answer pairs.
* **Key Columns:** `question`, `answer`, and `source`.

**Data Suitability & Limitations:**
* **Suitability:** The structured Q&A format is ideal for a retrieval-based system.
* **Limitations:** The dataset is domain-specific (cancer only), contains medical jargon requiring specialized preprocessing, and is static, meaning it does not update automatically with new medical discoveries.

## 4. Data Preprocessing

Effective data preprocessing was critical to transform raw text into a clean, consistent, and machine-readable format.

* **Initial Data Cleaning (Handling Duplicates):**
    * We identified and removed duplicate question-answer pairs, reducing the dataset from 729 to 683 unique entries. This prevents bias and ensures a precise knowledge base.
* **Text Preprocessing (Normalizing Questions and Answers):**
    * We developed a custom `clean_text` function, tailored for medical text. This pipeline included:
        * Removing HTML tags and boilerplate phrases.
        * Utilizing medical-specific stopwords.
        * Lowercasing text while preserving hyphens in medical terms.
        * Tokenization and lemmatization to reduce words to their base forms.
    * **Dual Cleaning Strategy:**
        * For **Search Indexing (`question_cleaned_for_search`)**: The full, aggressive `clean_text` pipeline was applied to the `question` column, optimizing it for TF-IDF vectorization and BioBERT embedding generation.
        * For **Display Readability (`answer_display`)**: A lighter cleaning process (removing newlines and collapsing spaces) was applied to the `answer` column, ensuring retrieved answers are human-readable and natural in the chatbot interface.

## 5. Modelling: Text Representation

To enable our chatbot to "understand" and compare text, we converted our cleaned questions into numerical representations using two distinct modeling approaches:

* **Lexical Modelling: TF-IDF (Term Frequency-Inverse Document Frequency)**
    * **Concept:** TF-IDF quantifies the importance of a word within a document relative to the entire collection. It assigns higher scores to words that are specific to a document.
    * **Role:** Facilitates direct keyword matching. It's highly effective for queries requiring precise term overlap.
    * **Implementation:** We used `TfidfVectorizer` on our `question_cleaned_for_search` column to create a sparse matrix (`X_tfidf`) representing each question's keyword profile.

* **Semantic Modelling: BioBERT Embeddings**
    * **Concept:** BioBERT (`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`) is a specialized transformer model trained on biomedical text. It generates dense numerical vectors (embeddings) that capture the contextual meaning and semantic relationships of words and sentences.
    * **Role:** Crucial for understanding the intent behind queries, even when exact keywords are not used (e.g., recognizing synonyms or paraphrases).
    * **Implementation:** We used the `SentenceTransformer` library to encode our `question_cleaned_for_search` into `question_embeddings`.

## 6. Hybrid Search System

Our `hybrid_search` system is the core intelligence that combines these two modeling approaches to retrieve the most relevant Q&A pairs.

* **Objective:** To accurately and intelligently retrieve information based on a user's query.
* **Approach:** Our `hybrid_search` function calculates both TF-IDF cosine similarity and BioBERT semantic cosine similarity between the user's query and all questions in our dataset.
* **Weighted Combination:** These two similarity scores are then combined using a weighted sum. This allows us to fine-tune the influence of lexical vs. semantic matching.
* **Confidence Threshold:** A configurable `threshold` is applied to the combined score; only matches exceeding this value are returned as "confident."
* **Disease Filtering:** An optional `disease_filter` allows for more precise, disease-specific searches.

## 7. Model Evaluation

We implemented a rigorous evaluation framework to assess our model's performance and guide optimization.

* **Validation Data:** We curated an expanded `validation_data` set, including diverse positive test cases and crucial negative (out-of-domain) test cases. Each test case included a `query`, `expected` keywords, an optional `disease` filter, and a `min_similarity` threshold.
* **`evaluate_model` Function:** Our custom `evaluate_model` function systematically processed each test case, checking for expected keywords in the retrieved answer and validating the similarity score against the `min_similarity` threshold. It provided detailed pass/fail status and reasons for failure.
* **Key Findings from Initial Evaluation:** Our initial evaluation on the expanded dataset yielded an accuracy of 71.43%. Detailed analysis of failures (e.g., high similarity with missing keywords, or unexpected similarity for out-of-domain queries) provided critical insights for improvement.

## 8. Optimization Results & Final Accuracy

To maximize performance, we conducted a two-stage hyperparameter tuning process:

* **1. Weight Tuning (TF-IDF vs. Semantic):**
    * **Process:** We systematically tested various combinations of `tfidf_weight` and `semantic_weight` (summing to 1.0) on our `validation_data`.
    * **Optimal Weights Found:** **TF-IDF Weight: 0.00**, **Semantic Weight: 1.00**.
    * **Impact:** This result strongly indicated that BioBERT's semantic understanding was overwhelmingly effective for our dataset, outperforming lexical matching alone or in combination.

* **2. Hybrid Search Internal Threshold Tuning:**
    * **Process:** With the optimal weights fixed (TF-IDF=0.00, Semantic=1.00), we then systematically tested a range of `threshold` values for the `hybrid_search` function's internal filtering.
    * **Optimal Threshold Found:** **0.30**.
    * **Impact:** This threshold proved crucial for effectively filtering out low-confidence, irrelevant matches, particularly for out-of-domain queries.

* **Final Achieved Accuracy:** After applying these optimized parameters (TF-IDF=0.00, Semantic=1.00, Threshold=0.30), our chatbot achieved a final accuracy of **85.71%** on our comprehensive validation set. This successfully addressed previous challenges, such as correctly identifying "No confident match found" for out-of-domain questions like "What is the best way to cook pasta?".

## 9. Challenges Encountered

Throughout the project, we navigated several significant challenges:

1.  **Data Quality and Preprocessing:** Handling heterogeneous XML formats, balancing aggressive cleaning for search with readability for display (leading to our dual-cleaning strategy), and managing duplicate entries were key hurdles.
2.  **Model Selection and Integration:** Effectively combining TF-IDF and BioBERT into a coherent hybrid system, and managing the computational demands of generating BioBERT embeddings, required careful design.
3.  **Performance Tuning and Evaluation:** Defining precise success criteria for evaluation, systematically optimizing multiple parameters (`weights`, `threshold`), and robustly handling edge cases (especially negative test cases) demanded iterative refinement. The initial "jibberish" output and `KeyError` issues during evaluation highlighted the need for meticulous debugging and robust error handling in our `evaluate_model` and `hybrid_search` functions.
4.  **User Interface Development:** Ensuring seamless integration between the backend Python logic and the Gradio frontend, and presenting cleaned data in a user-friendly format, required attention to detail.

## 10. User Interface

To make our optimized chatbot accessible and interactive, we developed a user-friendly web interface using **Gradio**.

* **Objective:** To provide an intuitive platform for users to query our medical information retrieval system.
* **Key Features:**
    * Simple text input for user questions.
    * Real-time interaction, leveraging our `chatbot_response` function which calls the optimized `hybrid_search`.
    * Formatted Markdown output, clearly displaying the closest matched question, the retrieved answer (display-friendly), and a confidence score.
    * Informative fallback messages when no confident match is found.
* **Benefits:** Offers ease of use, facilitates live demonstrations, and enables broader accessibility for testing and feedback.

## 11. Future Improvements & Recommendations

While our current prototype is robust, we envision several key areas for future development:

1.  **Enriching the Knowledge Base:** Expand and diversify the dataset with more comprehensive answers and additional reputable sources. Consider integrating content at a more granular level.
2.  **Advancing Retrieval and Generation:** Transition to a **Retrieval-Augmented Generation (RAG)** architecture, where retrieved passages are used by an LLM to *generate* comprehensive answers. Explore answer span extraction for more concise responses.
3.  **Enhancing User Interaction and Robustness:** Implement dynamic confidence management, "Did You Mean?" functionality for low-confidence queries, and a continuous user feedback loop for iterative improvement.
4.  **Operationalization and Scalability:** Develop a production-ready API (e.g., using FastAPI) and explore cloud deployment for scalability, reliability, and performance monitoring.
5.  **Ethical Considerations and Bias Mitigation:** Conduct data bias audits and explore methods for explainability to build user trust in this sensitive medical domain.

## 12. How to Run the Project Locally

To run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git # Replace with your actual repo URL
    cd your-repo-name # Navigate into the cloned directory
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt # If you have one, otherwise list them
    # Example if no requirements.txt:
    pip install pandas numpy scikit-learn nltk sentence-transformers gradio beautifulsoup4 lxml
    ```
    * **NLTK Data:** Ensure you download necessary NLTK data:
        ```python
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        ```
4.  **Download MedQuAD Dataset:**
    * The MedQuAD dataset needs to be downloaded. You'll typically place the XML files in a `data/MedQuAD` directory within your project. (Provide specific download instructions or a link if available, e.g., from a public dataset source).
5.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    * Open your main project notebook (e.g., `cancer_chatbot_project.ipynb`).
    * Run all cells in the notebook sequentially. This will load data, preprocess it, initialize models, define the hybrid search, run evaluations, and finally launch the Gradio interface.
6.  **Interact with the Chatbot:**
    * Once the Gradio cell runs, a local URL (e.g., `http://127.0.0.1:7860`) will appear in the notebook output. Open this link in your web browser to interact with the chatbot.

## 13. Contact / Contributors

* **[Group_8]** - [Contributers]
* Shallom Githui,JoyAran, SteveJoel, Simonrank, KemboiBett, Raphael
* Working UI: **https://v0-cancer-nlp-chatbot.vercel.app/**
