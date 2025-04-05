from xlm.components.generator.llm_generator import LLMGenerator
from xlm.dto.dto import ExplanationGranularity
from xlm.explainer.generic_generator_explainer import GenericGeneratorExplainer
from xlm.modules.registry.comparators import load_comparator
from xlm.modules.registry.perturbers import load_perturber

if __name__ == "__main__":
    perturber_name = "leave_one_out"
    comparator_name = "sentence_transformers_based_comparator"
    generator_model_name = "mistral-7b"
    lms_endpoint = "http://localhost:9985"

    user_input = """\
    Context: 
    The Panthers defense gave up just 308 points, ranking sixth in the league, while also leading the NFL in interceptions with 24 and boasting four Pro Bowl selections. Pro Bowl defensive tackle Kawann Short led the team in sacks with 11, while also forcing three fumbles and recovering two. Fellow lineman Mario Addison added 6½ sacks. The Panthers line also featured veteran defensive end Jared Allen, a 5-time pro bowler who was the NFL's active career sack leader with 136, along with defensive end Kony Ealy, who had 5 sacks in just 9 starts. Behind them, two of the Panthers three starting linebackers were also selected to play in the Pro Bowl: Thomas Davis and Luke Kuechly. Davis compiled 5½ sacks, four forced fumbles, and four interceptions, while Kuechly led the team in tackles (118) forced two fumbles, and intercepted four passes of his own. Carolina's secondary featured Pro Bowl safety Kurt Coleman, who led the team with a career high seven interceptions, while also racking up 88 tackles and Pro Bowl cornerback Josh Norman, who developed into a shutdown corner during the season and had four interceptions, two of which were returned for touchdowns.

    Question:
    How many points did the Panthers defense surrender?

    Answer: 
    """

    granularity = ExplanationGranularity.SENTENCE_LEVEL

    perturber = load_perturber(perturber_name=perturber_name)
    comparator = load_comparator(comparator_name=comparator_name)

    generator = LLMGenerator(endpoint=lms_endpoint, model_name=generator_model_name)
    reference_text = generator.generate(texts=[user_input])

    explainer = GenericGeneratorExplainer(
        perturber=perturber,
        generator=generator,
        comparator=comparator,
    )

    explanation_dto = explainer.explain(
        user_input=user_input,
        reference_text=reference_text[0],
        reference_score=None,
        granularity=granularity,
        do_normalize_comparator_scores=True,
    )

    print(explanation_dto.model_dump_json(exclude_defaults=True, indent=4))
