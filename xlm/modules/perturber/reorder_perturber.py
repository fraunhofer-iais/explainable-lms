from typing import List
import nlpaug.augmenter.word as naw
from xlm.modules.perturber.perturber import Perturber


class ReorderPerturber(Perturber):
    def __init__(self):
        self.__augmentor = naw.RandomWordAug(action="swap", aug_p=1.0)

    def perturb(self, text: str, features: List[str]) -> List[str]:
        perturbations = []
        for feature in features:
            augmented_text = self.__augmentor.augment(feature)
            perturbations.append(text.replace(feature, augmented_text[0]))
        return perturbations
