"""Missing step fields are observation gaps, never application progress."""

from gpu_fault.training_models import TrainingProgressHeartbeat, TrainingProgressState

LAST_VALID_STEP = "_gfrs_last_valid_step"


def last_valid_step(heartbeat: TrainingProgressHeartbeat | None) -> int | None:
    if heartbeat is None:
        return None
    if heartbeat.step is not None:
        return heartbeat.step
    raw = heartbeat.labels.get(LAST_VALID_STEP)
    return int(raw) if raw is not None and raw.isdecimal() else None


def next_progress_state(
    previous: TrainingProgressState | None, progress: TrainingProgressHeartbeat
) -> TrainingProgressState:
    before = previous.heartbeat if previous is not None else None
    identity_changed = before is not None and (
        before.node_id,
        before.pod_uid,
        before.container_name,
    ) != (progress.node_id, progress.pod_uid, progress.container_name)
    old_step = None if identity_changed else last_valid_step(before)
    advanced = progress.step is not None and (
        old_step is None or progress.step > old_step
    )
    labels = {
        key: value for key, value in progress.labels.items() if key != LAST_VALID_STEP
    }
    # Preserve the last known counter only across missing readings, using the
    # existing labels wire shape so old stored rows remain readable.
    if progress.step is None and old_step is not None:
        labels[LAST_VALID_STEP] = str(old_step)
    return TrainingProgressState(
        heartbeat=progress.model_copy(update={"labels": labels}),
        last_progress_at=(
            progress.observed_at
            if previous is None or identity_changed or advanced
            else previous.last_progress_at
        ),
    )
