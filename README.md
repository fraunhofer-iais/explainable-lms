# 🦉 RAG-Ex: A generic framework for explaining Retrieval Augmented Generation

Recent advancement of large language models (LLMs) has led to their seamless adoption in real-world applications,
including Retrieval Augmented Generation (RAG) in Question Answering (QA) tasks. However, owing to their size and
complexity, LLMs offer little to no explanations for why they generate a response, given the input. Such a "black-box"
ascription of LLMs effectively reduces the trust and confidence of end users in the emerging LLM-based RAG systems. In
this work, we introduce RAG-Ex, a model- and language-agnostic explanation framework that presents approximate
explanations to the users revealing why the LLMs possibly generated a piece of text as a response, given the user input.
Our framework is compatible with both open-source and proprietary LLMs. We report the significance scores of the
approximated explanations from our generic explainer in both English and German QA tasks and also study their
correlation with the downstream performance of the LLMs. In the extensive user studies, we observed that our explainer
yields an F1-score of 76.9\% against the end user annotations and attains almost in-par performance with model-intrinsic
approaches.

![readme_banner.png](xlm%2Fui%2Fimages%2Freadme_banner.png)

Find the demo application [here](https://github.com/fraunhofer-iais/language-model-service).

## How to use RAG-Ex?

### Pre-requisites

We recommend Python version >= 3.8

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### Step 1: Start the Language Model Service (LMS)

- Follow the instructions to install the LMS by checking out [this repo](https://github.com/fraunhofer-iais/language-model-service).
- Ensure the service is running at port `9985` such that ``http://localhost:9985`` is available.


### Step 2: Start RAG-Ex

You can start the UI for RAG-Ex with: `python -m xlm.start`.

### Step 3: Use RAG-Ex

The most important settings are as follows:

#### Question

Type your input here or pick one of the examples above.

#### Explanation Granularity

This sets the granularity of the explanations. E.g. "WORD_LEVEL" splits the output into words and gives an importance
score for every word.
"SENTENCE_LEVEL" collects every sentence and its importance score.

#### Importance Bounds

The Feature importance is between 0.0 and 1.0.
The higher, the more important was the sentence / word / ... of the input to generate the output.
The bound settings determine, from which importance score a feature is highlighted.
E.g. keeping the default settings (Upper=85, Middle=75, Lower=10) means that
features with an importance higher than 0.85 are highlighted in green and
features with an importance higher than 0.75 are highlighted yellow.

#### Model

The model that is used for all text generations.
This model is then used to both generate the original answer and the answers of all perturbed inputs.

#### Perturber

Here, we determine how to perturb the input. If the granularity is SENTENCE_LEVEL, the following holds:

| Perturber             | Explanation                                                                        |
|-----------------------|------------------------------------------------------------------------------------|
| leave_one_out         | For every perturbed input, we leave one sentence out.                              |
| random_word_perturber | We insert different random words in and around the corresponding sentence.         |
| entity_perturber      | Each entity in the sentences is replaced by random words.                          |
| reorder_perturber     | The sentence is reordered.                                                         |
| antonym_perturber     | One or more words in each sentence is replaced with their antonyms, if they exist. |
| synonym_perturber     | Same as antonym_perturber, but we replace with synonyms.                           |

#### Comparator

The comparator determines, how the gap between the original output and the perturbed output(s) is measured.

| Comparator                             | Explanation                                                                                                                                    |
|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| sentence_transformers_based_comparator | We take the SBERT embeddings and do a cosine similarity.                                                                                       |
| levenshtein_comparator                 | This counts, how many deletions, insertions and substitution of words within a sentence have to<br>be done to turn sentence A into sentence B. |
| jaro_winkler_comparator                | Similar to the levenshtein_comparator, while rewarding prefixes, i.e. when two sequences have<br>the same start.                               |
| n_gram_comparator                      | The sentences are split into adjacent word chunks of length n. The more chunks match, the closer the<br>sentences are.                         |

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

