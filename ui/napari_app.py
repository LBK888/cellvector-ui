"""Single napari/PySide6 desktop entry for CellVector."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import tifffile

from cellvector.domain.models import Point
from cellvector.execution.base import BackendHealth, PredictRequest
from cellvector.execution.local import LocalExecutionBackend
from cellvector.execution.remote import RemoteExecutionBackend
from cellvector.inference.nnunet.contracts import ExperimentSpec, InferenceProfile
from cellvector.inference.nnunet.registry import ModelRegistry
from cellvector.review_queue.models import QueueOrigin, QueueState

from .controller import CellVectorController


class OfflineExecutionBackend:
    def health(self) -> BackendHealth:
        return BackendHealth(status="offline", detail="remote worker is not configured")

    def predict(self, request: PredictRequest):
        raise ConnectionError("remote worker is not configured")


def configured_remote_backend():
    url = os.environ.get("CELLVECTOR_SERVER_URL")
    token = os.environ.get("CELLVECTOR_API_TOKEN")
    if url and token:
        return RemoteExecutionBackend(url, token=token)
    return OfflineExecutionBackend()


def launch_ui() -> None:
    try:
        import napari
        from magicgui.widgets import (
            CheckBox,
            ComboBox,
            Container,
            FileEdit,
            Label,
            LineEdit,
            PushButton,
            SpinBox,
        )
    except ImportError as error:
        raise RuntimeError(
            'CellVector UI dependencies are missing; install with pip install -e ".[ui]"'
        ) from error

    viewer = napari.Viewer(title="CellVector")
    controller = CellVectorController(
        local=LocalExecutionBackend(),
        remote=configured_remote_backend(),
    )

    source = FileEdit(label="2D TIFF/PNG", mode="r")
    annotation_output = FileEdit(label="Annotation JSON", mode="w")
    svg_output = FileEdit(label="SVG export", mode="w")
    ai_label_map = FileEdit(label="AI label map TIFF", mode="r")
    model_registry = FileEdit(label="Model registry JSON", mode="r")
    model_id = LineEdit(label="Model UUID")
    ai_profile = ComboBox(
        label="AI profile",
        choices=["preview", "standard", "accuracy"],
        value="preview",
    )
    frame = SpinBox(label="Frame", min=0, max=0, value=0)
    method = ComboBox(
        label="Classical method",
        choices=["hessian_vector", "fiji_reconstruction"],
        value="hessian_vector",
    )
    status = Label(value=f"GPU worker: {controller.remote_status}")
    open_button = PushButton(text="Open image")
    analyze_button = PushButton(text="Run classical analysis")
    accept_button = PushButton(text="Accept all proposal paths")
    commit_button = PushButton(text="Commit microridge edits")
    save_button = PushButton(text="Save annotation JSON")
    export_button = PushButton(text="Export SVG")
    load_ai_button = PushButton(text="Load AI proposal")
    train_dataset_id = SpinBox(label="nnU-Net dataset ID", min=1, max=999, value=501)
    train_snapshot_hash = LineEdit(label="Snapshot SHA-256")
    train_architecture = ComboBox(
        label="Architecture",
        choices=["plainconv", "resenc_l"],
        value="plainconv",
    )
    train_smoke = CheckBox(label="Software smoke only", value=False)
    remote_job_id = LineEdit(label="Remote job UUID")
    submit_train_button = PushButton(text="Submit remote train contract")
    job_status_button = PushButton(text="Refresh job status")
    cancel_job_button = PushButton(text="Cancel job")
    list_models_button = PushButton(text="List worker models")
    promotion_model_id = LineEdit(label="Promotion model UUID")
    promotion_actor = LineEdit(label="Promotion actor")
    promotion_reason = LineEdit(label="Promotion reason")
    promote_model_button = PushButton(text="Promote selected model")
    wp3_dataset_registry = FileEdit(label="Dataset lineage registry JSON", mode="w")
    wp3_queue_registry = FileEdit(label="Review queue JSON", mode="w")
    wp3_queue_item_id = LineEdit(label="Review item UUID")
    wp3_queue_state = ComboBox(
        label="Queue state",
        choices=["all", *[state.value for state in QueueState]],
        value="all",
    )
    wp3_queue_origin = ComboBox(
        label="Queue origin",
        choices=["all", *[origin.value for origin in QueueOrigin]],
        value="all",
    )
    wp3_actor = LineEdit(
        label="Reviewer",
        value=os.environ.get("USERNAME", "reviewer"),
    )
    wp3_note = LineEdit(label="Review note")
    configure_wp3_button = PushButton(text="Configure WP3 registries")
    list_queue_button = PushButton(text="List open review items")
    claim_queue_button = PushButton(text="Claim review item")
    release_queue_button = PushButton(text="Release review item")
    resolve_queue_button = PushButton(text="Commit and resolve review item")
    dismiss_queue_button = PushButton(text="Dismiss review item")

    def open_source() -> None:
        if not source.value:
            status.value = "Choose a TIFF or PNG file."
            return
        controller.open_image(Path(source.value), frame_index=int(frame.value))
        frame.max = max(0, len(controller.frames) - 1)
        _sync_layers(viewer, controller)
        status.value = (
            f"Opened {len(controller.frames)} frame(s); GPU worker: "
            f"{controller.remote_status}"
        )

    def select_frame(value: int) -> None:
        if controller.frames:
            controller.select_frame(int(value))
            _sync_layers(viewer, controller)

    def analyze_source() -> None:
        proposal = controller.run_classical(str(method.value))
        _sync_layers(viewer, controller)
        status.value = f"Proposal contains {len(proposal.microridges)} microridge paths."

    def accept_all() -> None:
        if controller.proposal is None:
            status.value = "Run a proposal first."
            return
        controller.accept_selected(
            [ridge.id for ridge in controller.proposal.microridges],
            author=os.environ.get("USERNAME", "reviewer"),
        )
        _sync_layers(viewer, controller)
        status.value = "Accepted proposal into a new annotation revision."

    def commit_edits() -> None:
        if controller.document is None:
            status.value = "Open an image first."
            return
        cell_paths = [
            [Point(x=float(x), y=float(y)) for y, x in np.asarray(shape)]
            for shape in viewer.layers["Final cells"].data
        ]
        ridge_paths = [
            [Point(x=float(x), y=float(y)) for y, x in np.asarray(shape)]
            for shape in viewer.layers["Final microridges"].data
        ]
        if cell_paths or controller.document.cells:
            controller.replace_all_cells(
                cell_paths,
                author=os.environ.get("USERNAME", "reviewer"),
            )
        controller.replace_all_microridges(
            ridge_paths,
            author=os.environ.get("USERNAME", "reviewer"),
        )
        _sync_layers(viewer, controller)
        status.value = "Committed microridge edits as a new revision."

    def save_annotation() -> None:
        if not annotation_output.value:
            status.value = "Choose an annotation JSON output path."
            return
        commit_edits()
        controller.save_json(Path(annotation_output.value))
        status.value = f"Saved {annotation_output.value}"

    def export_annotation() -> None:
        if not svg_output.value:
            status.value = "Choose an SVG output path."
            return
        commit_edits()
        controller.export_svg(Path(svg_output.value))
        status.value = f"Exported {svg_output.value}"

    def load_ai() -> None:
        if not ai_label_map.value or not model_registry.value or not model_id.value:
            status.value = "Choose AI label map, model registry, and model UUID."
            return
        proposal = _load_ai_prediction(
            controller,
            Path(ai_label_map.value),
            Path(model_registry.value),
            str(model_id.value),
            str(ai_profile.value),
        )
        _sync_layers(viewer, controller)
        status.value = f"Loaded AI proposal with {len(proposal.microridges)} paths."

    def submit_remote_train() -> None:
        try:
            accepted = _submit_remote_training(
                controller.remote,
                dataset_id=int(train_dataset_id.value),
                architecture=str(train_architecture.value),
                snapshot_hash=str(train_snapshot_hash.value),
                software_smoke_test=bool(train_smoke.value),
            )
            remote_job_id.value = accepted["job_id"]
            status.value = f"Training job {accepted['job_id']}: {accepted['status']}"
        except Exception as error:
            status.value = f"Remote training unavailable: {error}"

    def refresh_remote_job() -> None:
        try:
            payload = controller.remote.job_status(str(remote_job_id.value))
            status.value = f"Training job {payload['job_id']}: {payload['status']}"
        except Exception as error:
            status.value = f"Could not read remote job: {error}"

    def cancel_remote_job() -> None:
        try:
            payload = controller.remote.cancel_job(str(remote_job_id.value))
            status.value = f"Training job {payload['job_id']}: {payload['status']}"
        except Exception as error:
            status.value = f"Could not cancel remote job: {error}"

    def list_remote_models() -> None:
        try:
            models = controller.remote.list_models()
            summary = ", ".join(
                f"{item['model_id']} ({item['state']})" for item in models
            ) or "none"
            status.value = f"Worker models: {summary}"
        except Exception as error:
            status.value = f"Could not list worker models: {error}"

    def promote_remote_model() -> None:
        try:
            promoted = controller.remote.promote_model(
                str(promotion_model_id.value),
                actor=str(promotion_actor.value),
                reason=str(promotion_reason.value),
            )
            status.value = f"Model {promoted['model_id']}: {promoted['state']}"
        except Exception as error:
            status.value = f"Could not promote model: {error}"

    def configure_wp3() -> None:
        if not wp3_dataset_registry.value or not wp3_queue_registry.value:
            status.value = "Choose dataset lineage and review queue registry paths."
            return
        try:
            controller.configure_wp3(
                Path(wp3_dataset_registry.value),
                Path(wp3_queue_registry.value),
            )
            status.value = "WP3 registries configured; AI service is not required."
        except Exception as error:
            status.value = f"Could not configure WP3 registries: {error}"

    def list_review_queue() -> None:
        try:
            selected_state = str(wp3_queue_state.value)
            state_filter = None if selected_state == "all" else QueueState(selected_state)
            items = controller.list_review_queue(state=state_filter)
            selected_origin = str(wp3_queue_origin.value)
            if selected_origin != "all":
                items = [
                    item for item in items if item.origin == QueueOrigin(selected_origin)
                ]
            open_items = [
                item for item in items if item.state.value in {"queued", "claimed"}
            ]
            if not open_items:
                status.value = "Review queue: no open items."
                return
            wp3_queue_item_id.value = str(open_items[0].item_id)
            status.value = "Review queue: " + ", ".join(
                f"{item.item_id} ({item.state.value}, {item.priority.total:.3f})"
                for item in open_items
            )
        except Exception as error:
            status.value = f"Could not list review queue: {error}"

    def transition_review_item(action: str) -> None:
        try:
            item_id = UUID(str(wp3_queue_item_id.value))
            actor = str(wp3_actor.value)
            note = str(wp3_note.value)
            if action == "claim":
                item = controller.claim_review_item(item_id, actor=actor)
            elif action == "release":
                item = controller.release_review_item(item_id, actor=actor, note=note)
            elif action == "resolve":
                commit_edits()
                item = controller.resolve_review_item(item_id, actor=actor, note=note)
            else:
                item = controller.dismiss_review_item(item_id, actor=actor, note=note)
            status.value = f"Review item {item.item_id}: {item.state.value}"
        except Exception as error:
            status.value = f"Could not {action} review item: {error}"

    open_button.changed.connect(lambda _value: open_source())
    frame.changed.connect(select_frame)
    analyze_button.changed.connect(lambda _value: analyze_source())
    accept_button.changed.connect(lambda _value: accept_all())
    commit_button.changed.connect(lambda _value: commit_edits())
    save_button.changed.connect(lambda _value: save_annotation())
    export_button.changed.connect(lambda _value: export_annotation())
    load_ai_button.changed.connect(lambda _value: load_ai())
    submit_train_button.changed.connect(lambda _value: submit_remote_train())
    job_status_button.changed.connect(lambda _value: refresh_remote_job())
    cancel_job_button.changed.connect(lambda _value: cancel_remote_job())
    list_models_button.changed.connect(lambda _value: list_remote_models())
    promote_model_button.changed.connect(lambda _value: promote_remote_model())
    configure_wp3_button.changed.connect(lambda _value: configure_wp3())
    list_queue_button.changed.connect(lambda _value: list_review_queue())
    claim_queue_button.changed.connect(
        lambda _value: transition_review_item("claim")
    )
    release_queue_button.changed.connect(
        lambda _value: transition_review_item("release")
    )
    resolve_queue_button.changed.connect(
        lambda _value: transition_review_item("resolve")
    )
    dismiss_queue_button.changed.connect(
        lambda _value: transition_review_item("dismiss")
    )

    controls = Container(
        widgets=[
            source,
            frame,
            method,
            open_button,
            analyze_button,
            accept_button,
            commit_button,
            annotation_output,
            save_button,
            svg_output,
            export_button,
            ai_label_map,
            model_registry,
            promotion_model_id,
            ai_profile,
            load_ai_button,
            status,
        ]
    )
    viewer.window.add_dock_widget(controls, name="CellVector")
    operations = Container(
        widgets=[
            train_dataset_id,
            train_snapshot_hash,
            train_architecture,
            train_smoke,
            submit_train_button,
            remote_job_id,
            job_status_button,
            cancel_job_button,
            list_models_button,
            model_id,
            promotion_actor,
            promotion_reason,
            promote_model_button,
        ]
    )
    viewer.window.add_dock_widget(operations, name="WP2 Training and Models")
    review = Container(
        widgets=[
            wp3_dataset_registry,
            wp3_queue_registry,
            configure_wp3_button,
            wp3_queue_state,
            wp3_queue_origin,
            list_queue_button,
            wp3_queue_item_id,
            wp3_actor,
            wp3_note,
            claim_queue_button,
            release_queue_button,
            resolve_queue_button,
            dismiss_queue_button,
        ]
    )
    viewer.window.add_dock_widget(review, name="WP3 Review Queue")
    napari.run()


def _load_ai_prediction(
    controller: CellVectorController,
    label_map_path: Path,
    registry_path: Path,
    model_id: str,
    profile: str,
):
    labels = np.asarray(tifffile.imread(label_map_path), dtype=np.uint8)
    model = ModelRegistry(registry_path).get(UUID(model_id))
    return controller.add_ai_prediction(
        labels,
        model=model,
        profile=InferenceProfile(profile),
    )


def _submit_remote_training(
    backend: Any,
    *,
    dataset_id: int,
    architecture: str,
    snapshot_hash: str,
    software_smoke_test: bool,
) -> dict[str, Any]:
    if not hasattr(backend, "submit_train"):
        raise RuntimeError("remote worker is not configured")
    spec = (
        ExperimentSpec.plainconv(dataset_id)
        if architecture == "plainconv"
        else ExperimentSpec.resenc_l(dataset_id)
    )
    return backend.submit_train(
        spec,
        snapshot_hash=snapshot_hash,
        software_smoke_test=software_smoke_test,
        idempotency_key=str(uuid4()),
    )


def _sync_layers(viewer: Any, controller: CellVectorController) -> None:
    if controller.frame is None or controller.document is None:
        return
    _set_image(viewer, "Original", controller.frame.array)
    classical_paths = (
        [_shape(ridge.points) for ridge in controller.classical_proposal.microridges]
        if controller.classical_proposal is not None
        else []
    )
    ai_paths = (
        [_shape(ridge.points) for ridge in controller.ai_proposal.microridges]
        if controller.ai_proposal is not None
        else []
    )
    final_paths = [_shape(ridge.points) for ridge in controller.document.microridges]
    cell_shapes = [_shape(cell.contour) for cell in controller.document.cells]
    boundary_shapes = [_shape(item.points) for item in controller.document.boundaries]
    _set_shapes(viewer, "Classical proposal", classical_paths, "#ffb000", "path")
    _set_shapes(viewer, "AI proposal", ai_paths, "#7e57c2", "path")
    _set_shapes(viewer, "Final cells", cell_shapes, "#00bcd4", "polygon")
    _set_shapes(viewer, "Final boundaries", boundary_shapes, "#00bcd4", "path")
    _set_shapes(viewer, "Final microridges", final_paths, "#ff2d95", "path")


def _shape(points) -> np.ndarray:
    return np.asarray([[point.y, point.x] for point in points], dtype=float)


def _set_image(viewer: Any, name: str, data: np.ndarray) -> None:
    if name in viewer.layers:
        viewer.layers[name].data = data
    else:
        viewer.add_image(data, name=name, colormap="gray")


def _set_shapes(
    viewer: Any,
    name: str,
    paths: list[np.ndarray],
    color: str,
    shape_type: str,
) -> None:
    if name in viewer.layers:
        viewer.layers[name].data = paths
    else:
        viewer.add_shapes(
            paths,
            name=name,
            shape_type=shape_type,
            edge_color=color,
            face_color="transparent",
            edge_width=1,
            ndim=2,
        )
