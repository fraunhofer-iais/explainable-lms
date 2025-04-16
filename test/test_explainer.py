# import pytest
# from dotenv import load_dotenv
# from requests import Session
# from xlm.dto.dto import ExplanationDto
# from xlm.explainer.aleph_alpha_explainer import AlephAlphaExplainer
# from xlm.components.generator.llm_generator import LLMGenerator
# from xlm.dto.dto import ExplanationGranularity
#
# load_dotenv()
#
#
# @pytest.fixture
# def generator_e2e():
#     return LLMGenerator(
#         session=Session(),
#         endpoint="http://localhost:9985",
#     )
#
#
# @pytest.fixture
# def aleph_alpha_explainer_e2e(generator_e2e):
#     return AlephAlphaExplainer(generator=generator_e2e)
#
#
# @pytest.fixture
# def user_input_e2e() -> str:
#     text = """Input Context:\nThe Panthers defense gave up just 308 points, ranking sixth in the league, while also leading the NFL in interceptions with 24 and boasting four Pro Bowl selections. Pro Bowl defensive tackle Kawann Short led the team in sacks with 11, while also forcing three fumbles and recovering two. Fellow lineman Mario Addison added 6½ sacks. The Panthers line also featured veteran defensive end Jared Allen, a 5-time pro bowler who was the NFL's active career sack leader with 136, along with defensive end Kony Ealy, who had 5 sacks in just 9 starts. Behind them, two of the Panthers three starting linebackers were also selected to play in the Pro Bowl: Thomas Davis and Luke Kuechly. Davis compiled 5½ sacks, four forced fumbles, and four interceptions, while Kuechly led the team in tackles (118) forced two fumbles, and intercepted four passes of his own. Carolina's secondary featured Pro Bowl safety Kurt Coleman, who led the team with a career high seven interceptions, while also racking up 88 tackles and Pro Bowl cornerback Josh Norman, who developed into a shutdown corner during the season and had four interceptions, two of which were returned for touchdowns.\nQuestion:\nHow many forced fumbles did Thomas Davis have?\nAnswer: """
#     return text
#
#
# @pytest.mark.e2etest
# def test_aleph_alpha_explainer_e2e(user_input_e2e, aleph_alpha_explainer_e2e):
#     results = aleph_alpha_explainer_e2e.explain(
#         user_input=user_input_e2e,
#         granularity=ExplanationGranularity.SENTENCE_LEVEL,
#         model_name="luminous-supreme-control",
#         normalize=True,
#     )
#     assert isinstance(results, ExplanationDto)
#     assert results.output_text is not None
#     explanations = results.explanations
#     explanations_sample = explanations[0]
#     assert explanations_sample.score >= 0
#     assert isinstance(explanations_sample.score, float)
#     assert isinstance(explanations_sample.feature, str)
#     assert len(explanations) == 9
