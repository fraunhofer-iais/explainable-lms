from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from requests import Session

from xlm.components.generator.generator import Generator
from xlm.modules.registry import DEFAULT_LMS_ENDPOINT


class LLMGenerator(Generator):
    def __init__(
        self,
        model_name: str,
        session: Session = Session(),
        endpoint: str = DEFAULT_LMS_ENDPOINT,
        max_new_tokens: int = 100,
        split_lines: bool = True,
        temperature: float = 0,
        frequency_penalty: float = 2.0,
        presence_penalty: float = 2.0,
        num_threads: int = 10,
    ):
        self.__session = session
        self.__endpoint = endpoint
        self.__max_new_tokens = max_new_tokens
        self.__split_lines = split_lines
        self.__temperature = temperature
        self.__frequency_penalty = frequency_penalty
        self.__presence_penalty = presence_penalty
        self.__num_threads = num_threads
        self.model_name = model_name

    def generate(self, texts: List[str]) -> List[str]:
        if self.__num_threads == 1:
            return [
                self.__generate(text=text, model_name=self.model_name) for text in texts
            ]
        else:
            return self.__generate_multi_thread(texts=texts, model_name=self.model_name)

    def __generate(
        self,
        text: str,
        model_name: str,
    ) -> str:
        response = self.__session.post(
            url=f"{self.__endpoint}/generate",
            params={
                "model_name": model_name,
                "max_new_tokens": self.__max_new_tokens,
                "split_lines": self.__split_lines,
                "temperature": self.__temperature,
                "frequency_penalty": self.__frequency_penalty,
                "presence_penalty": self.__presence_penalty,
            },
            json=[text],
        )

        if response.status_code == 200:
            result = response.json()
            return result[0]
        else:
            raise ValueError(response.json())

    def __generate_multi_thread(self, texts: List[str], model_name: str) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.__num_threads) as executor:
            futures = [
                executor.submit(self.__run_generator, input_text, model_name)
                for input_text in texts
            ]
        result = [future.result() for future in as_completed(futures)]
        responses_dict = {
            list(res.items())[0][0]: list(res.items())[0][1] for res in result
        }
        responses = self.__collate(responses_dict=responses_dict, inputs=texts)
        return responses

    def __run_generator(self, input_text: str, model_name: str) -> Dict[str, str]:
        response = self.__generate(text=input_text, model_name=model_name)
        return {input_text: response}

    def __collate(
        self,
        responses_dict: Dict[str, str],
        inputs: List[str],
    ):
        return [responses_dict[inp] for inp in inputs]
