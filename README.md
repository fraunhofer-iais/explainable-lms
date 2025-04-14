# 🦉 RAG-Ex 2.0: Towards End-to-End Model-Agnostic Explanations for RAG Systems

RAG systems aim at improving response generation of Large Language Models (LLMs). 
With the help of the context provided by the user as input prompt, these systems are capable of generating more reliable responses. A typical RAG system is composed of a retriever in conjunction with a generator. Given a user question $q$, the retriever returns the most relevant documents $d_i$ from a collection. These documents together with an instruction composes the prompt $x$ which is then fed to the LLM-based generator. 
The generator finally returns a more reliable response $y$ to the user. This response is more reliable than the LLM generating an answer from its model weights alone. 
However, since the models used are not intrinsically explainable, end-users often find such RAG systems less trustworthy. 
In this work, we borrow ideas presented in our earlier works attempting to individually explain the retriever and the generator; and combine these strategies to build a holistic end-to-end framework towards model agnostic explanations for RAG systems. Our framework can explain retrievers (utilizing dense embedding models) and generators of any kind in an open-book QA setup. 


## Change log

- **v2.0:** Added retriever explanations, added interface for building RAG System. Find the release here.
- **v1.0:** Added generator explanations. Find the release [here](https://github.com/fraunhofer-iais/explainable-lms/releases/tag/v1.0).

## How to use RAG-Ex?

### Pre-requisites

We recommend Python version >= 3.8

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### Step 1: Start the Language Model Service (LMS)

- Follow the instructions to install the LMS by checking
  out [this repo](https://github.com/fraunhofer-iais/language-model-service).
- Ensure the service is running at port `9985` such that ``http://localhost:9985`` is available.

### Step 2: Start RAG-Ex

You can start the UI for RAG-Ex with: `python -m xlm.start`.

### Step 3: Use RAG-Ex

#### Examples
- To run generator explanations alone, follow the example [here](https://github.com/fraunhofer-iais/explainable-lms/blob/53d4eb0456e37bd91a9901d414d150a624ea69b0/examples/run_generic_generator_explainer.py).
- To run retriever explanations alone, follow the example [here](https://github.com/fraunhofer-iais/explainable-lms/blob/53d4eb0456e37bd91a9901d414d150a624ea69b0/examples/run_generic_retriever_explainer.py).
- To compose a RAG system and run retriever and generator explanations, follow the example [here](https://github.com/fraunhofer-iais/explainable-lms/blob/9805c0847b5fe2284e2523991935ea70d9c7932e/examples/run_rag_system.py). 

 
In the UI, follow the settings below:

#### Question

Type your input here or pick one of the examples above.

#### Explanation Granularity

This sets the granularity of the explanations. E.g. `WORD_LEVEL` splits the output into words and gives an importance
score for every word.
`SENTENCE_LEVEL` collects every sentence and its importance score.

#### Importance Bounds

The Feature importance is between 0.0 and 1.0.
The higher, the more important was the sentence / word / ... of the input to generate the output.
The bound settings determine, from which importance score a feature is highlighted.
E.g. keeping the default settings (`Upper`=85, `Middle`=75, `Lower`=10) means that
features with an importance higher than 0.85 are highlighted in green and
features with an importance higher than 0.75 are highlighted yellow.

#### Model

The model that is used for all text generations.
This model is then used to both generate the original answer and the answers of all perturbed inputs.

#### Perturber

Here, we determine how to perturb the input. If the granularity is `SENTENCE_LEVEL`, the following holds:

| Perturber               | Explanation                                                                        |
|-------------------------|------------------------------------------------------------------------------------|
| `leave_one_out`         | For every perturbed input, we leave one sentence out.                              |
| `random_word_perturber` | We insert different random words in and around the corresponding sentence.         |
| `entity_perturber`      | Each entity in the sentences is replaced by random words.                          |
| `reorder_perturber`     | The sentence is reordered.                                                         |
| `antonym_perturber`     | One or more words in each sentence is replaced with their antonyms, if they exist. |
| `synonym_perturber`     | Same as antonym_perturber, but we replace with synonyms.                           |

#### Comparator

The comparator determines how the gap between the original output and the perturbed output(s) is measured.

| Comparator                               | Explanation                                                                                                                                    |
|------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `sentence_transformers_based_comparator` | We take the SBERT embeddings and do a cosine similarity.                                                                                       |
| `levenshtein_comparator`                 | This counts, how many deletions, insertions and substitution of words within a sentence have to<br>be done to turn sentence A into sentence B. |
| `jaro_winkler_comparator`                | Similar to the levenshtein_comparator, while rewarding prefixes, i.e. when two sequences have<br>the same start.                               |
| `n_gram_comparator`                      | The sentences are split into adjacent word chunks of length n. The more chunks match, the closer the<br>sentences are.                         |

## Citation
