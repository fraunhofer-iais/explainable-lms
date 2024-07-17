from xlm.utils.categorizer import Categorizer
from xlm.dto.dto import ExplanationDto


UPPER_COLOR = "#D4EFDF"  # green
MID_COLOR = "#FBFBB8BF"  # amber
LOW_COLOR = "black"


class Visualizer:
    def __init__(self, show_mid_features: bool = True, show_low_features: bool = True):
        self.__show_mid_features = show_mid_features
        self.__show_low_features = show_low_features

    def visualize(
        self,
        segregator: Categorizer,
        explanations: ExplanationDto,
        output_from_explanations: str,
        avoid_exp_label: bool = False,
    ) -> str:
        highlighted_text = output_from_explanations

        pos_features, mid_features, low_features = segregator.categorize(
            explanations=explanations
        )

        if not self.__show_mid_features:
            mid_features = []

        if not self.__show_low_features:
            low_features = []

        for explanation in explanations.explanations:
            score = round(explanation.score, 2)

            if score == 0:
                continue

            if explanation.feature in pos_features:
                token_str = (
                    '<span title="'
                    + str(score)
                    + '"style="font-weight:bold;background-color:'
                    + UPPER_COLOR
                    + '">'
                    + explanation.feature
                    + "</span>"
                )
            elif explanation.feature in mid_features:
                token_str = (
                    '<span title="'
                    + str(score)
                    + '"style="font-weight:bold;background-color:'
                    + MID_COLOR
                    + '">'
                    + explanation.feature
                    + "</span>"
                )
            else:
                token_str = (
                    '<span title="'
                    + str(score)
                    + '"style="color:'
                    + LOW_COLOR
                    + '">'
                    + explanation.feature
                    + "</span>"
                )

            highlighted_text = highlighted_text.replace(explanation.feature, token_str)

        if avoid_exp_label:
            vis = "<p>" + highlighted_text + "</p>"
        else:
            vis = "<p><b>Explanations:</b><br>" + highlighted_text + "</p>"
        vis = vis.replace("\n", "<br>")

        legend = "<p align='right'"

        legend += (
            '<span title="' + '"style="color:' + LOW_COLOR + '">' + "💡" + "</span>"
        )

        legend += "&emsp;"

        legend += (
            '<span title="'
            + '"style="color:'
            + LOW_COLOR
            + '">'
            + "not important"
            + "</span>"
        )

        legend += "&emsp;⇢&emsp;"

        legend += (
            '<span title="'
            + '"style="font-weight:bold;background-color:'
            + MID_COLOR
            + '">'
            + " important "
            + "</span>"
        )

        legend += "&emsp;⇢&emsp;"

        legend += (
            '<span title="'
            + '"style="font-weight:bold;background-color:'
            + UPPER_COLOR
            + '">'
            + " very important "
            + "</span>"
        )

        legend += "</p>"

        html_str = legend + vis

        return html_str
