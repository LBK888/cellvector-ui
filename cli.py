"""CellVector command-line interface."""

from __future__ import annotations

from enum import StrEnum
import json
from pathlib import Path
from uuid import UUID, uuid4

import typer

from cellvector.application.datasets import DatasetEntry, create_dataset_snapshot
from cellvector.agreement.metrics import compare_annotations
from cellvector.datasets.diff import compare_snapshots
from cellvector.datasets.lineage import DatasetRegistry
from cellvector.datasets.models import DatasetSnapshot
from cellvector.datasets.nnunet import export_nnunet_dataset
from cellvector.domain.models import AnnotationDocument
from cellvector.domain.validation import validate_document
from cellvector.execution.base import PredictRequest
from cellvector.execution.local import LocalExecutionBackend
from cellvector.execution.remote import RemoteExecutionBackend
from cellvector.export.svg import export_svg
from cellvector.inference.nnunet.commands import build_plan_command, build_train_command
from cellvector.inference.nnunet.contracts import ExperimentSpec
from cellvector.inference.nnunet.environment import check_ai_environment
from cellvector.extensions.contracts import (
    SyntheticAdapterManifest,
    SyntheticAllowedUse,
    TopologyExperimentManifest,
    validate_synthetic_use,
    validate_topology_activation,
)
from cellvector.regression.evaluate import evaluate_policy
from cellvector.regression.models import RegressionEvidence, RegressionPolicy
from cellvector.review_queue.models import QueueOrigin, QueuePriorityPolicy, QueueState
from cellvector.review_queue.registry import ReviewQueueRegistry


class ClassicalMethod(StrEnum):
    FIJI = "fiji_reconstruction"
    HESSIAN = "hessian_vector"


class AIArchitecture(StrEnum):
    PLAINCONV = "plainconv"
    RESENC_L = "resenc_l"


app = typer.Typer(
    name="cellvector",
    no_args_is_help=True,
    help="Vector-first cell membrane and microridge annotation.",
)
dataset_app = typer.Typer(help="Create immutable dataset snapshots and nnU-Net exports.")
train_app = typer.Typer(help="Submit and manage remote training jobs.")
models_app = typer.Typer(help="Inspect and manually promote registered models.")
agreement_app = typer.Typer(help="Compare independent reviewed annotations.")
queue_app = typer.Typer(help="Manage the local auditable WP3 review queue.")
regression_app = typer.Typer(help="Evaluate versioned regression policies.")
extensions_app = typer.Typer(help="Validate disabled-by-default WP3 extensions.")
app.add_typer(dataset_app, name="dataset")
app.add_typer(train_app, name="train")
app.add_typer(models_app, name="models")
app.add_typer(agreement_app, name="agreement")
app.add_typer(queue_app, name="review-queue")
app.add_typer(regression_app, name="regression")
app.add_typer(extensions_app, name="extensions")


def _write_json_model(value, output: Path | None) -> None:
    serialized = value.model_dump_json(indent=2)
    if output is None:
        typer.echo(serialized)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    typer.echo(f"Wrote {output}")


@dataset_app.command(name="register")
def dataset_register_command(
    snapshot_path: Path,
    registry_path: Path = typer.Option(..., "--registry"),
    actor: str = typer.Option(..., "--actor"),
    parent_hash: str | None = typer.Option(None, "--parent-hash"),
) -> None:
    snapshot = DatasetSnapshot.model_validate_json(snapshot_path.read_text(encoding="utf-8"))
    record = DatasetRegistry(registry_path).register(
        snapshot, snapshot_path, actor=actor, parent_hash=parent_hash
    )
    typer.echo(record.model_dump_json(indent=2))


@dataset_app.command(name="diff")
def dataset_diff_command(
    base: Path,
    target: Path,
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    before = DatasetSnapshot.model_validate_json(base.read_text(encoding="utf-8"))
    after = DatasetSnapshot.model_validate_json(target.read_text(encoding="utf-8"))
    _write_json_model(compare_snapshots(before, after), output)


@agreement_app.command(name="compare")
def agreement_compare_command(
    annotations: list[Path] = typer.Argument(...),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    documents = [
        AnnotationDocument.model_validate_json(path.read_text(encoding="utf-8"))
        for path in annotations
    ]
    _write_json_model(compare_annotations(documents), output)


@queue_app.command(name="list")
def queue_list_command(
    registry_path: Path = typer.Option(..., "--registry"),
    state: QueueState | None = typer.Option(None, "--state"),
) -> None:
    items = ReviewQueueRegistry(registry_path).list(state=state)
    typer.echo(json.dumps([item.model_dump(mode="json") for item in items], indent=2))


@queue_app.command(name="add")
def queue_add_command(
    source_sha256: str,
    frame_index: int = typer.Option(0, "--frame", min=0),
    registry_path: Path = typer.Option(..., "--registry"),
    origin: QueueOrigin = typer.Option(QueueOrigin.MANUAL),
    priority: float = typer.Option(1.0, min=0, max=1),
    reason: str = typer.Option("manual-review", "--reason"),
) -> None:
    item = ReviewQueueRegistry(registry_path).add(
        source_sha256=source_sha256,
        frame_index=frame_index,
        origin=origin,
        priority=QueuePriorityPolicy.manual_only().score({"manual": priority}),
        reasons=(reason,),
    )
    typer.echo(item.model_dump_json(indent=2))


def _queue_transition(
    registry_path: Path,
    item_id: UUID,
    action: str,
    actor: str,
    note: str,
    revision: UUID | None = None,
):
    registry = ReviewQueueRegistry(registry_path)
    if action == "claim":
        return registry.claim(item_id, actor=actor)
    if action == "release":
        return registry.release(item_id, actor=actor, note=note)
    if action == "resolve":
        return registry.resolve(
            item_id,
            actor=actor,
            result_revision_id=revision,
            note=note,
        )
    return registry.dismiss(item_id, actor=actor, note=note)


def _emit_queue_transition(
    item_id: UUID,
    registry_path: Path,
    actor: str,
    note: str,
    action: str,
) -> None:
    item = _queue_transition(registry_path, item_id, action, actor, note)
    typer.echo(item.model_dump_json(indent=2))


@queue_app.command(name="claim")
def queue_claim_command(
    item_id: UUID,
    registry_path: Path = typer.Option(..., "--registry"),
    actor: str = typer.Option(..., "--actor"),
    note: str = typer.Option("", "--note"),
) -> None:
    _emit_queue_transition(item_id, registry_path, actor, note, "claim")


@queue_app.command(name="release")
def queue_release_command(
    item_id: UUID,
    registry_path: Path = typer.Option(..., "--registry"),
    actor: str = typer.Option(..., "--actor"),
    note: str = typer.Option("", "--note"),
) -> None:
    _emit_queue_transition(item_id, registry_path, actor, note, "release")


@queue_app.command(name="dismiss")
def queue_dismiss_command(
    item_id: UUID,
    registry_path: Path = typer.Option(..., "--registry"),
    actor: str = typer.Option(..., "--actor"),
    note: str = typer.Option("", "--note"),
) -> None:
    _emit_queue_transition(item_id, registry_path, actor, note, "dismiss")


@queue_app.command(name="resolve")
def queue_resolve_command(
    item_id: UUID,
    result_revision_id: UUID = typer.Option(..., "--result-revision"),
    registry_path: Path = typer.Option(..., "--registry"),
    actor: str = typer.Option(..., "--actor"),
    note: str = typer.Option("", "--note"),
) -> None:
    typer.echo(_queue_transition(registry_path, item_id, "resolve", actor, note, result_revision_id).model_dump_json(indent=2))


@regression_app.command(name="evaluate")
def regression_evaluate_command(
    policy_path: Path,
    evidence_path: Path,
    metrics_path: Path,
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    policy = RegressionPolicy.model_validate_json(policy_path.read_text(encoding="utf-8"))
    evidence = RegressionEvidence.model_validate_json(evidence_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _write_json_model(evaluate_policy(policy, evidence, metrics), output)


@extensions_app.command(name="validate")
def extensions_validate_command(
    manifest_path: Path,
    kind: str = typer.Option(..., "--kind"),
    requested_use: SyntheticAllowedUse = typer.Option(SyntheticAllowedUse.SOFTWARE_FIXTURE, "--requested-use"),
) -> None:
    if kind == "topology":
        value = validate_topology_activation(TopologyExperimentManifest.model_validate_json(manifest_path.read_text(encoding="utf-8")))
        typer.echo(value.model_dump_json(indent=2))
        return
    manifest = SyntheticAdapterManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    validate_synthetic_use(manifest, requested_use)
    typer.echo(manifest.model_dump_json(indent=2))


def _remote(base_url: str, token: str) -> RemoteExecutionBackend:
    return RemoteExecutionBackend(base_url, token=token)


@dataset_app.command(name="snapshot")
def dataset_snapshot_command(
    annotations: list[Path] = typer.Argument(...),
    output: Path = typer.Option(..., "--output", "-o"),
    created_by: str = typer.Option(..., "--created-by"),
    seed: int = typer.Option(42, "--seed"),
    reviewed: bool = typer.Option(False, "--reviewed"),
    software_smoke_test: bool = typer.Option(False, "--software-smoke-test"),
) -> None:
    """Create a snapshot from explicitly review-complete annotation revisions."""

    if not reviewed:
        raise typer.BadParameter(
            "--reviewed is required to confirm these exact revisions are review-complete"
        )
    documents = [
        AnnotationDocument.model_validate_json(path.read_text(encoding="utf-8"))
        for path in annotations
    ]
    snapshot = create_dataset_snapshot(
        [DatasetEntry(document=document, reviewed=True) for document in documents],
        seed=seed,
        created_by=created_by,
        software_smoke_test=software_smoke_test,
        output_path=str(output.resolve()),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"Wrote {output} snapshot_hash={snapshot.identity_hash()}")


@dataset_app.command(name="export-nnunet")
def dataset_export_nnunet_command(
    snapshot_path: Path,
    annotations: list[Path] = typer.Argument(...),
    output_root: Path = typer.Option(..., "--output-root"),
    dataset_id: int = typer.Option(..., "--dataset-id", min=1, max=999),
    dataset_name: str = typer.Option("CellVector", "--dataset-name"),
) -> None:
    """Export one immutable snapshot to native 2D nnU-Net TIFF layout."""

    snapshot = DatasetSnapshot.model_validate_json(
        snapshot_path.read_text(encoding="utf-8")
    )
    documents = [
        AnnotationDocument.model_validate_json(path.read_text(encoding="utf-8"))
        for path in annotations
    ]
    report = export_nnunet_dataset(
        snapshot,
        {document.revision_id: document for document in documents},
        output_root,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
    )
    typer.echo(
        f"Wrote {report.dataset_root} ({len(report.artifact_sha256)} verified artifacts)"
    )


@train_app.command(name="submit")
def train_submit_command(
    dataset_id: int = typer.Argument(..., min=1, max=999),
    snapshot_hash: str = typer.Option(..., "--snapshot-hash"),
    architecture: AIArchitecture = typer.Option(AIArchitecture.PLAINCONV),
    base_url: str = typer.Option("http://127.0.0.1:8765", "--base-url"),
    token: str = typer.Option(..., "--token", envvar="CELLVECTOR_API_TOKEN"),
    idempotency_key: str | None = typer.Option(None, "--idempotency-key"),
    software_smoke_test: bool = typer.Option(False, "--software-smoke-test"),
) -> None:
    """Submit a reproducible train contract and verify the returned artifact."""

    spec = (
        ExperimentSpec.plainconv(dataset_id)
        if architecture == AIArchitecture.PLAINCONV
        else ExperimentSpec.resenc_l(dataset_id)
    )
    artifact = _remote(base_url, token).train_contract(
        spec,
        snapshot_hash=snapshot_hash,
        software_smoke_test=software_smoke_test,
        idempotency_key=idempotency_key or str(uuid4()),
    )
    typer.echo(json.dumps(artifact, indent=2, sort_keys=True))


@train_app.command(name="status")
def train_status_command(
    job_id: UUID,
    base_url: str = typer.Option("http://127.0.0.1:8765", "--base-url"),
    token: str = typer.Option(..., "--token", envvar="CELLVECTOR_API_TOKEN"),
) -> None:
    typer.echo(json.dumps(_remote(base_url, token).job_status(job_id), indent=2))


@train_app.command(name="cancel")
def train_cancel_command(
    job_id: UUID,
    base_url: str = typer.Option("http://127.0.0.1:8765", "--base-url"),
    token: str = typer.Option(..., "--token", envvar="CELLVECTOR_API_TOKEN"),
) -> None:
    typer.echo(json.dumps(_remote(base_url, token).cancel_job(job_id), indent=2))


@models_app.command(name="list")
def models_list_command(
    base_url: str = typer.Option("http://127.0.0.1:8765", "--base-url"),
    token: str = typer.Option("", "--token", envvar="CELLVECTOR_API_TOKEN"),
) -> None:
    typer.echo(json.dumps(_remote(base_url, token).list_models(), indent=2))


@models_app.command(name="promote")
def models_promote_command(
    model_id: UUID,
    actor: str = typer.Option(..., "--actor"),
    reason: str = typer.Option(..., "--reason"),
    base_url: str = typer.Option("http://127.0.0.1:8765", "--base-url"),
    token: str = typer.Option(..., "--token", envvar="CELLVECTOR_API_TOKEN"),
) -> None:
    promoted = _remote(base_url, token).promote_model(
        model_id,
        actor=actor,
        reason=reason,
    )
    typer.echo(json.dumps(promoted, indent=2))


@app.command()
def analyze(
    source: Path,
    output: Path = typer.Option(..., "--output", "-o"),
    method: ClassicalMethod = ClassicalMethod.HESSIAN,
    frame: int = typer.Option(0, min=0),
) -> None:
    """Run one classical pipeline and write an annotation proposal document."""

    result = LocalExecutionBackend().predict(
        PredictRequest(
            image_path=source,
            method=method.value,
            frame_index=frame,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.document.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"Wrote {output} ({len(result.proposal.microridges)} microridge paths)")


@app.command(name="validate")
def validate_command(annotation: Path) -> None:
    """Validate annotation schema and topology."""

    document = AnnotationDocument.model_validate_json(annotation.read_text(encoding="utf-8"))
    report = validate_document(document)
    if report.errors:
        for issue in report.errors:
            typer.echo(f"ERROR {issue.code}: {issue.message}")
        raise typer.Exit(code=1)
    typer.echo(f"Valid annotation; {len(report.warnings)} warning(s).")


@app.command(name="export-svg")
def export_svg_command(
    annotation: Path,
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Export a reviewed annotation as pixel-coordinate SVG."""

    document = AnnotationDocument.model_validate_json(annotation.read_text(encoding="utf-8"))
    export_svg(document, output)
    typer.echo(f"Wrote {output}")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Start the optional FastAPI reference worker."""

    import uvicorn

    from cellvector.server.app import create_app

    uvicorn.run(create_app(), host=host, port=port)


@app.command(name="ai-check")
def ai_check(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Report optional PyTorch, CUDA, and nnU-Net availability."""

    report = check_ai_environment()
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        return
    typer.echo(f"AI ready: {report.ready}")
    typer.echo(f"Python: {report.python_version}")
    typer.echo(f"PyTorch: {report.torch_version or 'not installed'}")
    typer.echo(f"nnU-Net: {report.nnunet_version or 'not installed'}")
    typer.echo(f"CUDA: {report.cuda_available}")
    for code in report.errors:
        typer.echo(f"ERROR {code}")


@app.command(name="nnunet-commands")
def nnunet_commands(
    dataset_id: int = typer.Argument(..., min=1, max=999),
    architecture: AIArchitecture = typer.Option(AIArchitecture.PLAINCONV),
    fold: int = typer.Option(0, min=0, max=4),
) -> None:
    """Preview validated nnU-Net argv without executing it."""

    spec = (
        ExperimentSpec.plainconv(dataset_id)
        if architecture == AIArchitecture.PLAINCONV
        else ExperimentSpec.resenc_l(dataset_id)
    )
    typer.echo(" ".join(build_plan_command(spec)))
    typer.echo(" ".join(build_train_command(spec, fold=fold)))


@app.command()
def ui() -> None:
    """Open the single CellVector napari desktop entry."""

    from cellvector.ui.napari_app import launch_ui

    launch_ui()


if __name__ == "__main__":
    app()
