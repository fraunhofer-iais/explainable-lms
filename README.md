# 🦉 RAG-Ex: A generic framework for explaining Retrieval Augmented Generation

Recent advancement of large language models (LLMs) has led to their seamless adoption in real-world applications, including Retrieval Augmented Generation (RAG) in Question Answering (QA) tasks. However, owing to their size and complexity, LLMs offer little to no explanations for why they generate a response, given the input. Such a "black-box" ascription of LLMs effectively reduces the trust and confidence of end users in the emerging LLM-based RAG systems. In this work, we introduce RAG-Ex, a model- and language-agnostic explanation framework that presents approximate explanations to the users revealing why the LLMs possibly generated a piece of text as a response, given the user input. Our framework is compatible with both open-source and proprietary LLMs. We report the significance scores of the approximated explanations from our generic explainer in both English and German QA tasks and also study their correlation with the downstream performance of the LLMs. In the extensive user studies, we observed that our explainer yields an F1-score of 76.9\% against the end user annotations and attains almost in-par performance with model-intrinsic approaches.

![readme_banner.png](xlm%2Fui%2Fimages%2Freadme_banner.png)

Find the demo application [here](https://huggingface.co/spaces/vijusudhi/rag-ex).

## How to use RAG-Ex?

### Pre-requisites

We recommend Python version >= 3.8

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### Step 1: Start the Language Model Service (LMS)

- Follow the instructions to install the LMS by checking out [this repo]().
- Ensure the service is running at port `9985` such that ``http://localhost:9985`` is available.

[comment]: <> (add link to lms repo)

### Step 2: Start RAG-Ex

You can start the UI for RAG-Ex with: `python -m xlm.start`.

## Citation

To use RAG-Ex in your publication, please cite it by using the following BibTeX entry.
```
@inproceedings{rag-ex,
    title = "RAG-Ex: A generic framework for explaining Retrieval Augmented Generation",
    author = "Sudhi, Viju and
      Bhat, Sinchana Ramakanth and
      Rudat, Max and
      Teucher, Roman",
    booktitle = "Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval July 2024",
    month = july,
    year = "2024",
    address = "Washington D.C., USA"
}
```

