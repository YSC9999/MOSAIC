"""Unit tests for the typed data contracts."""

import json
import math

import pytest

from mosaic.contracts import (
    Box,
    ClarityReport,
    ContractError,
    DecodeDiagnostics,
    Detection,
    MaskEvidence,
    MotionGateReport,
    ObjectEvidence,
    TrackSummary,
    TrackToken,
    WindowDiagnostics,
    WindowOutput,
)


# ---------------------------------------------------------------- helpers
def make_token(**overrides):
    values = dict(
        track_id=3,
        timestamp=0.5,
        frame_index=15,
        box=Box(10, 20, 50, 100),
        observed=True,
        detector_confidence=0.9,
        visibility_score=0.8,
        mask_available=False,
        centroid=[30.0, 60.0],
        velocity=[1.0, -2.0],
        acceleration=[0.0, 0.1],
    )
    values.update(overrides)
    return TrackToken(**values)


def make_window(**overrides):
    values = dict(
        window_id="w0",
        source_id="clip1.avi",
        start_timestamp=0.0,
        end_timestamp=5.0,
        label="violent",
        violence_score=0.8,
        temporal_confidence=0.6,
        clarity_score=0.7,
        processed_frame_count=40,
        skipped_frame_count=110,
    )
    values.update(overrides)
    return WindowOutput(**values)


# ------------------------------------------------------------------- Box
def test_valid_box_properties():
    box = Box(10, 20, 50, 100)
    assert box.width == 40 and box.height == 80 and box.area == 3200
    assert box.centroid == [30.0, 60.0]
    assert box.to_list() == [10.0, 20.0, 50.0, 100.0]
    assert Box.from_list(box.to_list()) == box


@pytest.mark.parametrize(
    "coords",
    [
        (10, 10, 5, 20),  # x2 < x1
        (10, 10, 20, 5),  # y2 < y1
        (10, 10, 10, 20),  # zero width
        (10, 10, 20, 10),  # zero height
        (-1, 0, 5, 5),  # negative coordinate
        (0, 0, float("nan"), 5),
        (0, 0, float("inf"), 5),
        ("a", 0, 5, 5),
        (True, 0, 5, 5),  # bool is not a number here
    ],
)
def test_malformed_boxes_are_rejected(coords):
    with pytest.raises(ContractError):
        Box(*coords)


def test_box_must_stay_inside_frame():
    box = Box(10, 10, 50, 50)
    box.check_within(100, 100)
    box.check_within(50, 50)  # touching the edge is fine
    with pytest.raises(ContractError):
        box.check_within(49, 100)


@pytest.mark.parametrize("bad", [[1, 2, 3], [1, 2, 3, 4, 5], "abcd", None, [1, 2, 3, "x"]])
def test_box_from_list_rejects_malformed_input(bad):
    with pytest.raises(ContractError):
        Box.from_list(bad)


# ------------------------------------------------------------- Detection
def test_detection_roundtrip():
    detection = Detection(Box(1, 1, 5, 9), 0.5, "person", 4, 0.13, pooled_feature=[0.1, 0.2])
    assert Detection.from_dict(detection.to_dict()) == detection


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf"), "0.5", None, True])
def test_detection_confidence_is_range_checked(confidence):
    with pytest.raises(ContractError):
        Detection(Box(1, 1, 5, 9), confidence, "person", 0, 0.0)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 0.5])
def test_detection_confidence_edges_are_allowed(confidence):
    Detection(Box(1, 1, 5, 9), confidence, "person", 0, 0.0)


def test_detection_rejects_bad_fields():
    with pytest.raises(ContractError):
        Detection(Box(1, 1, 5, 9), 0.5, "", 0, 0.0)  # empty class name
    with pytest.raises(ContractError):
        Detection(Box(1, 1, 5, 9), 0.5, "person", -1, 0.0)  # negative index
    with pytest.raises(ContractError):
        Detection(Box(1, 1, 5, 9), 0.5, "person", 0, -0.1)  # negative time
    with pytest.raises(ContractError):
        Detection([1, 1, 5, 9], 0.5, "person", 0, 0.0)  # box must be a Box
    with pytest.raises(ContractError):
        Detection(Box(1, 1, 5, 9), 0.5, "person", 0, 0.0, pooled_feature=[])
    with pytest.raises(ContractError):
        Detection(Box(1, 1, 5, 9), 0.5, "person", 0, 0.0, pooled_feature=[math.nan])


def test_from_dict_reports_missing_key():
    data = Detection(Box(1, 1, 5, 9), 0.5, "person", 0, 0.0).to_dict()
    del data["confidence"]
    with pytest.raises(ContractError, match="confidence"):
        Detection.from_dict(data)


# --------------------------------------------------------------- Clarity
def test_clarity_marks_unavailable_components_explicitly():
    report = ClarityReport(score=0.6, source_profile="480p", resolution_component=0.5)
    assert "person_scale_component" in report.unavailable_components
    assert "resolution_component" not in report.unavailable_components
    assert ClarityReport.from_dict(report.to_dict()) == report


@pytest.mark.parametrize("score", [-0.1, 1.5, float("nan")])
def test_clarity_score_range(score):
    with pytest.raises(ContractError):
        ClarityReport(score=score, source_profile="x")


def test_clarity_component_range():
    with pytest.raises(ContractError):
        ClarityReport(score=0.5, source_profile="x", sharpness_component=2.0)


# ------------------------------------------------ Object / mask evidence
def test_object_evidence_roundtrip_and_validation():
    item = ObjectEvidence(1, "stick", Box(1, 1, 4, 4), 0.4, 7, associated_track_id=2)
    assert ObjectEvidence.from_dict(item.to_dict()) == item
    with pytest.raises(ContractError):
        ObjectEvidence(1, "stick", Box(1, 1, 4, 4), 1.2, 7)
    with pytest.raises(ContractError):
        ObjectEvidence(1, "", Box(1, 1, 4, 4), 0.5, 7)
    with pytest.raises(ContractError):
        ObjectEvidence(1, "stick", Box(1, 1, 4, 4), 0.5, 7, associated_track_id=-3)


def test_mask_visibility_fraction_is_range_checked():
    MaskEvidence("mask://1", 0.9, 0.0, 3)
    MaskEvidence("mask://1", 0.9, 1.0, 3)
    for fraction in (-0.1, 1.1, float("nan")):
        with pytest.raises(ContractError):
            MaskEvidence("mask://1", 0.9, fraction, 3)
    item = MaskEvidence("mask://1", 0.9, 0.5, 3, contour_features=[1.0, 2.0])
    assert MaskEvidence.from_dict(item.to_dict()) == item


# ------------------------------------------------------------ TrackToken
def test_observed_token_roundtrip():
    token = make_token(
        clarity_score=0.7,
        mask_available=True,
        silhouette_features=[0.1, 0.2, 0.3],
        object_evidence=[ObjectEvidence(0, "stick", Box(1, 1, 3, 3), 0.5, 15)],
    )
    assert TrackToken.from_dict(token.to_dict()) == token


def test_predicted_token_has_no_detector_confidence():
    predicted = make_token(observed=False, detector_confidence=None)
    assert predicted.observed is False
    assert predicted.detector_confidence is None


def test_observed_token_requires_confidence():
    with pytest.raises(ContractError):
        make_token(observed=True, detector_confidence=None)


def test_predicted_token_must_not_fake_a_confidence():
    with pytest.raises(ContractError):
        make_token(observed=False, detector_confidence=0.9)


def test_silhouette_requires_mask_available():
    with pytest.raises(ContractError):
        make_token(mask_available=False, silhouette_features=[0.1])


@pytest.mark.parametrize(
    "field,value",
    [
        ("track_id", -1),
        ("timestamp", -1.0),
        ("visibility_score", 1.5),
        ("centroid", [1.0]),
        ("velocity", [1.0, 2.0, 3.0]),
        ("acceleration", [float("nan"), 0.0]),
        ("observed", 1),
        ("clarity_score", 2.0),
        ("box", [1, 2, 3, 4]),
    ],
)
def test_malformed_tokens_are_rejected(field, value):
    with pytest.raises(ContractError):
        make_token(**{field: value})


# ---------------------------------------------------------- TrackSummary
def test_track_summary_cannot_end_before_start():
    with pytest.raises(ContractError):
        TrackSummary(1, start_timestamp=2.0, end_timestamp=1.0, observed_count=1, predicted_count=0)
    summary = TrackSummary(1, 0.0, 2.0, 10, 2, mean_detector_confidence=0.7)
    assert TrackSummary.from_dict(summary.to_dict()) == summary


# ------------------------------------------------------ DecodeDiagnostics
def good_decode(**overrides):
    values = dict(
        source_id="a.avi", source_kind="file", status="ok", decoded_frame_count=10,
        fps=30.0, width=64, height=48,
    )
    values.update(overrides)
    return DecodeDiagnostics(**values)


def test_decode_diagnostics_roundtrip():
    diag = good_decode(declared_frame_count=10, first_timestamp=0.0, last_timestamp=0.3)
    assert DecodeDiagnostics.from_dict(diag.to_dict()) == diag


def test_decode_failure_must_explain_itself():
    with pytest.raises(ContractError):
        DecodeDiagnostics("a.avi", "file", "failed", 0)
    DecodeDiagnostics("a.avi", "file", "failed", 0, errors=["cannot open"])


def test_decode_ok_cannot_have_errors_or_zero_frames():
    with pytest.raises(ContractError):
        good_decode(errors=["boom"])
    with pytest.raises(ContractError):
        good_decode(decoded_frame_count=0)


def test_decode_empty_requires_zero_frames():
    with pytest.raises(ContractError):
        DecodeDiagnostics("a.avi", "file", "empty", 3, errors=["x"])


@pytest.mark.parametrize("fps", [0, -5, float("nan")])
def test_decode_rejects_invalid_fps(fps):
    with pytest.raises(ContractError):
        good_decode(fps=fps)


def test_decode_unknown_status_or_kind():
    with pytest.raises(ContractError):
        good_decode(status="great")
    with pytest.raises(ContractError):
        good_decode(source_kind="telepathy")


# ------------------------------------------------- WindowDiagnostics etc.
def test_window_diagnostics_indices_must_be_ordered_and_disjoint():
    WindowDiagnostics(selected_frame_indices=[0, 2, 4], skipped_frame_indices=[1, 3])
    with pytest.raises(ContractError):
        WindowDiagnostics(selected_frame_indices=[2, 1])
    with pytest.raises(ContractError):
        WindowDiagnostics(selected_frame_indices=[1, 1])
    with pytest.raises(ContractError):
        WindowDiagnostics(selected_frame_indices=[1, 2], skipped_frame_indices=[2, 3])
    with pytest.raises(ContractError):
        WindowDiagnostics(selected_frame_indices=[-1])


# ---------------------------------------------------------- WindowOutput
def test_window_output_roundtrip_through_json():
    window = make_window(
        tracks=[TrackSummary(1, 0.0, 4.0, 100, 5, 0.8)],
        objects=[ObjectEvidence(0, "stick", Box(1, 1, 3, 3), 0.4, 10)],
        diagnostics=WindowDiagnostics(
            selected_frame_indices=[0, 5], skipped_frame_indices=[1, 2],
            decode=good_decode(), motion_scores=[0.1, 0.9], notes=["hello"],
        ),
    )
    restored = WindowOutput.from_json(window.to_json())
    assert restored == window
    assert json.loads(window.to_json())["label"] == "violent"


@pytest.mark.parametrize("label", ["goon", "VIOLENT", "", None, "instigator"])
def test_window_label_is_only_violent_or_non_violent(label):
    with pytest.raises(ContractError):
        make_window(label=label)


@pytest.mark.parametrize("field", ["violence_score", "temporal_confidence", "clarity_score"])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_window_scores_are_range_checked(field, value):
    with pytest.raises(ContractError):
        make_window(**{field: value})


def test_window_cannot_end_before_it_starts():
    with pytest.raises(ContractError):
        make_window(start_timestamp=3.0, end_timestamp=2.0)


def test_window_requires_clarity_score_not_a_default():
    # clarity_score has no default value: forgetting it is an error, not a silent 0.
    with pytest.raises(TypeError):
        WindowOutput(
            window_id="w", source_id="s", start_timestamp=0, end_timestamp=1, label="violent",
            violence_score=0.5, temporal_confidence=0.5, processed_frame_count=1,
            skipped_frame_count=0,
        )


def test_window_output_has_no_role_or_instigator_fields():
    keys = set(make_window().to_dict())
    assert not any("instigator" in k or "goon" in k or "role" in k for k in keys)


def test_window_from_dict_requires_diagnostics_key():
    data = make_window().to_dict()
    del data["diagnostics"]
    with pytest.raises(ContractError, match="diagnostics"):
        WindowOutput.from_dict(data)


def test_motion_gate_report_contract_validation():
    rep = MotionGateReport(
        total_frames=100,
        selected_frames=30,
        skipped_frames=70,
        reduction_fraction=0.70,
        mean_motion_score=0.04,
        burst_count=5,
        keep_alive_count=4,
    )
    assert MotionGateReport.from_dict(rep.to_dict()) == rep

    # Selected + skipped must equal total
    with pytest.raises(ContractError, match="total"):
        MotionGateReport(
            total_frames=100,
            selected_frames=20,
            skipped_frames=70,
            reduction_fraction=0.70,
            mean_motion_score=0.04,
            burst_count=5,
            keep_alive_count=4,
        )

    # Reduction fraction must be in [0, 1]
    with pytest.raises(ContractError, match="reduction_fraction"):
        MotionGateReport(
            total_frames=100,
            selected_frames=30,
            skipped_frames=70,
            reduction_fraction=1.5,
            mean_motion_score=0.04,
            burst_count=5,
            keep_alive_count=4,
        )

