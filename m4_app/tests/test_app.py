"""M4 contract tests; no private evaluation files required."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import pytest
from fastapi.testclient import TestClient

from backend.config import ROOT, Settings
from backend.dataset import InvalidTrip, load_trip
from backend.main import create_app
from backend.scoring import collapse_consecutive, score_prediction


@pytest.fixture
def demo_client(tmp_path):
    settings = Settings(mode="demo", demo_dir=ROOT / "demo_data",
                        log_path=tmp_path / "requests.jsonl")
    with TestClient(create_app(settings)) as client:
        yield client, settings


def test_demo_is_explicitly_marked_and_two_trips_are_listed(demo_client):
    client, _ = demo_client
    health = client.get('/api/health')
    assert health.status_code == 200
    assert health.json()['mode'] == 'demo'
    assert health.json()['status'] == 'ready'
    resp = client.get('/api/trips').json()
    assert resp['count'] == 2
    assert [x['id'] for x in resp['trips']] == ['demo_01', 'demo_02']


def test_predictions_have_road_geometry_and_server_scoring(demo_client):
    client, settings = demo_client
    response = client.post('/api/predict', json={'trip_id': 'demo_01'})
    assert response.status_code == 200, response.text
    doc = response.json()
    assert doc['predictor'] == 'geometry_demo_baseline'
    assert doc['observation_count'] == len(doc['predicted_route']['arc_ids_per_observation'])
    assert len(doc['gps_points']) == doc['observation_count']
    assert len(doc['predicted_route']['segments']) >= 1
    assert doc['evaluation']['available'] is True
    assert 'route_accuracy_percent' in doc['evaluation']
    body = response.text
    # No raw reference arc list or raw route file content may be present.
    assert 'reference_route' not in doc
    assert 'actual_route' not in doc
    assert 'reference_arc_ids' not in doc
    loglines = settings.log_path.read_text().strip().splitlines()
    record = json.loads(loglines[-1])
    assert record['path'] == '/api/predict'
    assert record['trip_id'] == 'demo_01'
    assert record['status'] == 200
    assert 'ground_truth' not in record


def test_input_validation_and_no_directory_traversal(demo_client):
    client, _ = demo_client
    assert client.post('/api/predict', json={'trip_id': '../demo_01'}).status_code == 422
    assert client.post('/api/predict', json={'trip_id': 'missing_trip'}).status_code == 404


def test_real_mode_requires_model_and_real_data(tmp_path):
    folder = tmp_path / 'evaluator'
    folder.mkdir()
    for ext in ('.nodes', '.arcs', '.track', '.route'):
        shutil.copyfile(ROOT / 'demo_data' / ('demo_01' + ext), folder / ('eval_01' + ext))
    settings = Settings(mode='trained', eval_dir=folder,
                        model_path=tmp_path/'missing.joblib', log_path=tmp_path/'req.jsonl')
    with TestClient(create_app(settings)) as client:
        health = client.get('/api/health').json()
        assert health['status'] == 'not_ready'
        result = client.post('/api/predict', json={'trip_id': 'eval_01'})
        assert result.status_code == 503
        assert 'Missing final trained model' in result.json()['detail']


def test_score_respects_order_and_no_pointwise_claim():
    assert collapse_consecutive([0, 0, 1, 1, 2]) == [0, 1, 2]
    good = score_prediction([0, 0, 1, 2], [0, 1, 2])
    bad = score_prediction([2, 1, 0], [0, 1, 2])
    assert good['route_accuracy_percent'] == 100.0
    assert bad['route_accuracy_percent'] < good['route_accuracy_percent']
    assert 'top1_accuracy_percent' not in good  # no per-point claim without the trip geometry


def test_observation_top1_uses_aligned_reference_server_side():
    from backend.dataset import load_trip
    from backend.scoring import load_reference_private
    trip = load_trip(ROOT / 'demo_data', 'demo_01')
    reference = load_reference_private(ROOT / 'demo_data', 'demo_01', len(trip.arcs))
    # Predicting the aligned reference arc for every point must score 100 %.
    import numpy as np, sys
    sys.path.insert(0, str(ROOT.parent))
    from m1_data.labels import align_route
    from m1_data.loader import Trip
    from m1_data.network import RoadNetwork
    m1 = Trip('demo_01', 'evaluator_test',
              np.array([[o.lon, o.lat, o.timestamp] for o in trip.observations]),
              np.asarray(trip.nodes, dtype=float), np.asarray(trip.arcs, dtype=np.int64))
    net = RoadNetwork(m1)
    px, py = net.proj.to_xy(m1.lon, m1.lat)
    aligned, _, _ = align_route(net, px, py, np.asarray(reference))
    perfect = score_prediction([int(a) for a in aligned], reference, trip)
    assert perfect['top1_accuracy_percent'] == 100.0
    wrong = [int((a + 1) % len(trip.arcs)) for a in aligned]
    assert score_prediction(wrong, reference, trip)['top1_accuracy_percent'] < 100.0


def test_all_ten_trip_ids_can_be_served_when_dataset_exists(tmp_path):
    # Contract test only: these are synthetic copies of a fixture, NOT final evaluator results.
    for i in range(10):
        for ext in ('.nodes', '.arcs', '.track', '.route'):
            shutil.copyfile(ROOT/'demo_data'/('demo_02'+ext), tmp_path/f'case_{i:02}{ext}')
    settings = Settings(mode='demo', demo_dir=tmp_path, log_path=tmp_path/'logs'/'req.jsonl')
    with TestClient(create_app(settings)) as client:
        result = client.get('/api/trips').json()
        assert result['count'] == 10
        for trip in result['trips']:
            response = client.post('/api/predict', json={'trip_id': trip['id']})
            assert response.status_code == 200, response.text
            assert response.json()['trip_id'] == trip['id']


def test_ground_truth_missing_returns_no_fake_score(tmp_path):
    for ext in ('.nodes','.arcs','.track'):
        shutil.copyfile(ROOT/'demo_data'/('demo_01'+ext),tmp_path/('unlabelled'+ext))
    settings = Settings(mode='demo', demo_dir=tmp_path, log_path=tmp_path/'log.jsonl')
    with TestClient(create_app(settings)) as client:
        response = client.post('/api/predict', json={'trip_id': 'unlabelled'})
        assert response.status_code == 200
        assert response.json()['evaluation']['available'] is False
        assert 'route_accuracy_percent' not in response.json()['evaluation']


def test_invalid_dataset_rejected(tmp_path):
    (tmp_path/'broken.track').write_text('77.60 12.95 0\n77.60 12.95 -1\n')
    (tmp_path/'broken.nodes').write_text('77.60 12.95\n')
    (tmp_path/'broken.arcs').write_text('0 0\n')
    with pytest.raises(InvalidTrip):
        load_trip(tmp_path, 'broken')


def test_nested_trip_folder_layout_supported(tmp_path):
    child = tmp_path/'case_nested'
    child.mkdir()
    for ext in ('.nodes', '.arcs', '.track', '.route'):
        shutil.copyfile(ROOT/'demo_data'/('demo_01'+ext), child/('actual_name'+ext))
    settings = Settings(mode='demo', demo_dir=tmp_path, log_path=tmp_path/'req.jsonl')
    with TestClient(create_app(settings)) as client:
        assert client.get('/api/trips').json()['trips'] == [{'id': 'case_nested'}]
        assert client.post('/api/predict',json={'trip_id':'case_nested'}).status_code == 200
