import re
from typing import List
import spacy
from lingua import Language, LanguageDetectorBuilder
from xlm.modules.tokenizer.tokenizer import Tokenizer
from xlm.dto.dto import ExplanationGranularity


class CustomTokenizer(Tokenizer):
    def __init__(self):
        self.__detector = LanguageDetectorBuilder.from_languages(
            *[Language.ENGLISH, Language.GERMAN]
        ).build()
        self.__en_nlp = spacy.load("en_core_web_sm")
        self.__de_nlp = spacy.load("de_core_news_sm")

    def tokenize(self, text: str, granularity: ExplanationGranularity) -> List[str]:
        lang = self.__detector.detect_language_of(text=text)
        nlp = self.__en_nlp if lang == Language.ENGLISH else self.__de_nlp

        if granularity == ExplanationGranularity.WORD_LEVEL:
            return self.__word_tokenize(text=text, nlp=nlp)
        elif granularity == ExplanationGranularity.SENTENCE_LEVEL:
            return self.__sent_tokenize(text=text, nlp=nlp)
        elif granularity == ExplanationGranularity.PARAGRAPH_LEVEL:
            return self.__paragraph_tokenize(text=text, nlp=nlp)
        elif granularity == ExplanationGranularity.PHRASE_LEVEL:
            return self.__phrase_tokenize(text=text, nlp=nlp)
        else:
            raise ValueError("Incorrect granularity level passed!")

    def __word_tokenize(self, text: str, nlp) -> List[str]:
        tokens = []
        doc = nlp(text)
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            tokens.append(token.text)
        tokens = [token.strip() for token in tokens]
        tokens = [token for token in tokens if token]
        return tokens

    def __sent_tokenize(self, text: str, nlp) -> List[str]:
        paragraphs = self.__paragraph_tokenize(text=text, nlp=nlp)
        tokens = []
        for paragraph in paragraphs:
            doc = nlp(paragraph)
            for sent in doc.sents:
                tokens.append(sent.text)
        tokens = [token.strip() for token in tokens]
        tokens = [token for token in tokens if token]
        return tokens

    def __paragraph_tokenize(self, text: str, nlp) -> List[str]:
        text = re.sub("\n+", "\n", text)
        tokens = [p for p in text.split("\n") if p]
        tokens = sorted(set(tokens), key=tokens.index)
        tokens = [token.strip() for token in tokens]
        tokens = [token for token in tokens if token]
        return tokens

    def __phrase_tokenize(self, text: str, nlp) -> List[str]:
        doc = nlp(text)
        tokens = []
        for noun_chunk in doc.noun_chunks:
            if nlp(noun_chunk.text)[0].is_stop or nlp(noun_chunk.text)[0].is_punct:
                continue
            tokens.append(noun_chunk.text)
        tokens = [token.strip() for token in tokens]
        tokens = [token for token in tokens if token]
        return tokens
