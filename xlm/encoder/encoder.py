from typing import List
from requests import Session

from xlm.registry import DEFAULT_LMS_ENDPOINT


class Encoder:
    def __init__(
        self,
        model_name: str,
        session: Session = Session(),
        endpoint: str = DEFAULT_LMS_ENDPOINT,
    ):
        self.__session = session
        self.__endpoint = endpoint
        self.__model_name = model_name

    def encode(self, texts: List[str]) -> List[List[float]]:
        response = self.__session.post(
            url=f"{self.__endpoint}/vectorize",
            params={
                "model_name": self.__model_name,
            },
            json=texts,
        )
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            raise ValueError(f"{self.__model_name} is not en encoder model!")
