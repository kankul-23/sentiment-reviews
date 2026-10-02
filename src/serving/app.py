"""REST API: python -m uvicorn src.serving.app:app --port 8000

GET  /health         живость и готовность модели
GET  /info           модель, температура калибровки, max_length, устройство
POST /predict        {"text": "..."} -> оценка 1-5 и откалиброванные вероятности
POST /predict/batch  {"texts": [...]} -> список предсказаний в том же порядке (до 64 текстов)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src.serving.schemas import BatchPrediction, BatchRequest, PredictRequest, Prediction

VERSION = "0.1.0"


def create_app(model_loader=None) -> FastAPI:
    """model_loader: функция без аргументов, возвращающая модель (в тестах подставляется заглушка)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if model_loader is not None:
            app.state.model = model_loader()
        else:
            from src.serving.inference import SentimentModel
            from src.serving.settings import load_settings

            app.state.model = SentimentModel.load(load_settings())
        yield

    app = FastAPI(title="sentiment-reviews", version=VERSION, lifespan=lifespan,
                  description="Оценка (1-5) по тексту русскоязычного отзыва на организацию.")

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_loaded": getattr(request.app.state, "model", None) is not None}

    @app.get("/info")
    def info(request: Request):
        return {"version": VERSION, **request.app.state.model.info(),
                "calibration": "temperature scaling, T подобрана на val"}

    @app.post("/predict", response_model=Prediction)
    def predict(req: PredictRequest, request: Request):
        return request.app.state.model.predict([req.text])[0]

    @app.post("/predict/batch", response_model=BatchPrediction)
    def predict_batch(req: BatchRequest, request: Request):
        return {"predictions": request.app.state.model.predict(req.texts)}

    return app


app = create_app()
