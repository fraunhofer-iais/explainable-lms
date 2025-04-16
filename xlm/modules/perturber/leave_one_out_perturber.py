from typing import List
from xlm.modules.perturber.perturber import Perturber


class LeaveOneOutPerturber(Perturber):
    def perturb(self, text: str, features: List[str]) -> List[str]:
        perturbations = []
        for token in features:
            perturbations.append(
                # re.sub(token, "", text).strip()
                text.replace(token, "").strip()
            )
        return perturbations
