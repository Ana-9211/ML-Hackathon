from pathlib import Path
from backend.config import ROOT, Settings
from backend.dataset import load_trip
from backend.pipeline import CandidateIndex, _emissions, viterbi, prediction_geometry


def test_one_arc_per_observation_and_road_geometry():
    trip = load_trip(ROOT/'demo_data', 'demo_02')
    idx = CandidateIndex(trip)
    candidates = [idx.top_candidates(i) for i in range(len(trip.observations))]
    assert all(1 <= len(group) <= 6 for group in candidates)
    emissions = _emissions(trip, candidates, Settings(mode='demo'))
    predicted = viterbi(trip, candidates, emissions)
    assert len(predicted) == len(trip.observations)
    route = prediction_geometry(trip, predicted)
    assert len(route['segments']) == len(route['ordered_arcs'])
    assert 0 <= route['continuity_percent'] <= 100


def test_trained_mode_accepts_a_team_generated_joblib_binary_candidate_model(tmp_path):
    """Toy artifact checks wiring only; no real training results or evaluator data."""
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from fastapi.testclient import TestClient
    from backend.main import create_app
    from backend.pipeline import _make_features
    trip = load_trip(ROOT/'demo_data', 'demo_01')
    index = CandidateIndex(trip)
    groups = [index.top_candidates(i) for i in range(len(trip.observations))]
    features = _make_features(trip, groups)
    # Arbitrary toy labels only; NOT ground-truth or a meaningful trained M3 model.
    toy_labels = np.arange(len(features)) % 2
    toy_model = LogisticRegression(max_iter=500).fit(features, toy_labels)
    path = tmp_path/'toy_model.joblib'
    joblib.dump(toy_model, path)
    settings = Settings(mode='trained', eval_dir=ROOT/'demo_data', model_path=path,
                        log_path=tmp_path/'log.jsonl')
    with TestClient(create_app(settings)) as client:
        result = client.post('/api/predict', json={'trip_id':'demo_01'})
        assert result.status_code == 200, result.text
        assert result.json()['predictor'] == 'joblib_m3_plus_viterbi'
