import sys

from ui.explainer_ui import ExplainerUI
from visualizer.visualizer import Visualizer


def load_visualizer() -> Visualizer:
    visualizer = Visualizer(show_mid_features=True, show_low_features=True)
    return visualizer


if __name__ == "__main__":
    interface = ExplainerUI(
        logo_path="ui/images/iais.svg",
        css_path="ui/css/demo.css",
        visualizer=load_visualizer(),
        window_title="RAG-Ex",
        title="✳️ RAG-Ex: Towards Generic Explainability",
    )
    app = interface.build_app()
    app.queue()

    platform = sys.platform
    print("Platform: ", platform)
    host = "127.0.0.1" if "win" in platform else "0.0.0.0"
    port = 8001
    app.launch(server_name=host, server_port=port)
