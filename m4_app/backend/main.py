"""PathSnap M4: FastAPI runs on Laptop 1; Leaflet UI runs on Laptop 2."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import logging
import time
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import Settings
from .dataset import InvalidTrip, list_trip_ids, load_trip, safe_trip_id
from .pipeline import PipelineUnavailable, predict_trip, prediction_geometry
from .scoring import load_reference_private, score_prediction


class PredictionRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=120)


def _logger(settings: Settings) -> logging.Logger:
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    name = "pathsnap_request_log_" + uuid.uuid4().hex
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(settings.log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    log = _logger(settings)
    app = FastAPI(title="PathSnap M4 Map Matching API", version="1.0.0",
                  description="All inference and reference scoring happen server-side on Laptop 1.")
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.origins),
                       allow_credentials=False, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        request.state.trip_id = None
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            response.headers["Cache-Control"] = "no-store"
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            line = {
                "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": status_code,
                "duration_ms": round((time.perf_counter()-started)*1000, 2),
                "client_ip": request.client.host if request.client else None,
                "trip_id": request.state.trip_id,
                # Important: no raw GPS or ground-truth arc IDs in logs
            }
            log.info(json.dumps(line, ensure_ascii=True))
            print("M4 REQUEST", json.dumps(line, ensure_ascii=True), flush=True)

    @app.get("/api/health")
    def health():
        trips = list_trip_ids(settings.data_dir)
        model_ready = (settings.mode == "demo" or bool(settings.team_predictor)
                       or settings.model_path.is_file())
        if trips:
            data_hint = f"Found {len(trips)} compatible trip(s)."
        elif settings.mode == "demo":
            data_hint = "Included demo trips are missing; restore the demo_data folder from the package."
        else:
            data_hint = "No trips found. Set M4_EVAL_DIR to a folder containing matching .track, .nodes and .arcs files."
        return {
            "service": "pathsnap-m4", "mode": settings.mode,
            "status": "ready" if trips and model_ready else "not_ready",
            "available_trip_count": len(trips), "model_configured": model_ready,
            "data_ready": bool(trips),
            "data_location": "demo_data" if settings.mode == "demo" else "M4_EVAL_DIR",
            "data_hint": data_hint,
            "note": "Demo mode uses synthetic trips and is not an evaluator submission" if settings.mode == "demo" else "No reference data is sent to inference or UI",
        }

    @app.get("/api/trips")
    def trips():
        # Only IDs, never reference paths, hidden labels, or source filenames.
        ids = list_trip_ids(settings.data_dir)
        return {"count": len(ids), "trips": [{"id": trip_id} for trip_id in ids],
                "mode": settings.mode}

    @app.post("/api/predict")
    def predict(payload: PredictionRequest, request: Request):
        try:
            trip_id = safe_trip_id(payload.trip_id)
        except InvalidTrip as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        request.state.trip_id = trip_id
        try:
            trip = load_trip(settings.data_dir, trip_id, settings.max_points, settings.max_arcs)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Unknown trip ID") from exc
        except InvalidTrip as exc:
            raise HTTPException(status_code=422, detail=f"Trip input validation failed: {exc}") from exc
        try:
            predictions, source = predict_trip(trip, settings)
        except PipelineUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (ValueError, ArithmeticError) as exc:
            raise HTTPException(status_code=422, detail=f"Trip cannot be decoded: {exc}") from exc
        geometry = prediction_geometry(trip, predictions)
        # Scoring is strictly after inference; only this function accesses .route.
        try:
            actual_route = load_reference_private(settings.data_dir, trip_id, len(trip.arcs))
            evaluation = score_prediction(predictions, actual_route)
        except InvalidTrip as exc:
            raise HTTPException(status_code=500, detail=f"Private reference file is invalid: {exc}") from exc
        return {
            "trip_id": trip_id,
            "mode": settings.mode,
            "predictor": source,
            "observation_count": len(trip.observations),
            "gps_points": [[p.lat, p.lon] for p in trip.observations],
            "predicted_route": geometry,
            "evaluation": evaluation,
            # Return NO route ground-truth IDs and NO *.route contents.
        }

    return app


app = create_app()
