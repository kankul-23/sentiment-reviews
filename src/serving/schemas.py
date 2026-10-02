from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, field_validator

MAX_TEXT_CHARS = 20_000
MAX_BATCH = 64

Text = Annotated[str, StringConstraints(min_length=1, max_length=MAX_TEXT_CHARS)]


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("текст не должен состоять из одних пробелов")
    return value


class PredictRequest(BaseModel):
    text: Text

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        return _not_blank(value)


class BatchRequest(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=MAX_BATCH)

    @field_validator("texts")
    @classmethod
    def _each_not_blank(cls, values: list[str]) -> list[str]:
        for v in values:
            _not_blank(v)
        return values


class Prediction(BaseModel):
    rating: int = Field(ge=1, le=5, description="предсказанная оценка (класс с максимальной вероятностью)")
    confidence: float = Field(description="откалиброванная вероятность предсказанного класса")
    expected_rating: float = Field(description="ожидаемая оценка по откалиброванным вероятностям")
    probabilities: dict[str, float] = Field(description="откалиброванные вероятности классов 1-5")
    truncated: bool = Field(description="текст длиннее max_length токенов и был обрезан")


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
