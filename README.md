# 🦉 RAG-Ex: A generic framework for explaining Retrieval Augmented Generation

Welcome to our XLM (eXplainable Language Model Service) Repository.

Find the demo application [here](https://huggingface.co/spaces/vijusudhi/rag-ex).

![readme_banner.png](xlm%2Fui%2Fimages%2Freadme_banner.png)

## Set-Up

### XLM

Install dependencies

```
pip install -r requirements.txt
```

Download Spacy Packages

```
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

Run UI

```
python -m xlm.start.py
```

### LMS

In order to use our LMS (Language Model Service), you need to 
1. clone and run [this repo]()
2. make it reachable from "http://localhost:9985" (which should be the default)

[comment]: <> (add link to running mock demo)
[comment]: <> (add link to lms repo)
