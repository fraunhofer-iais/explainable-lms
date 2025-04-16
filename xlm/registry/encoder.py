from xlm.components.encoder.encoder import Encoder
from xlm.registry import DEFAULT_LMS_ENDPOINT


def load_encoder(model_name: str, endpoint: str = DEFAULT_LMS_ENDPOINT):
    return Encoder(
        model_name=model_name,
        endpoint=endpoint,
    )
