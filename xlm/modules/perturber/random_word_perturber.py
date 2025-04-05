from typing import List
import nltk
import numpy as np
from nltk.corpus import wordnet
from xlm.modules.perturber.perturber import Perturber

nltk.download("wordnet")
np.random.seed(42)


class RandomWordPerturber(Perturber):
    def perturb(self, text: str, features: List[str]) -> List[str]:
        num_words_to_add = 2
        perturbations = []
        for feature in features:
            words = feature.split()
            random_words = [
                self.__get_random_word_from_wordnet() for _ in range(num_words_to_add)
            ]
            for _ in range(num_words_to_add):
                position = np.random.randint(0, len(words) + 1)
                words.insert(position, random_words.pop())
            noisy_sentence = " ".join(words)
            perturbations.append(text.replace(feature, noisy_sentence))
        return perturbations

    def __get_random_word_from_wordnet(self):
        synsets = list(wordnet.all_synsets())
        random_synset = np.random.choice(synsets)
        random_word = np.random.choice(random_synset.lemma_names())
        return random_word
