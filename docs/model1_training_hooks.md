# Part C: Model 1 training hooks

`fit()` accepts two optional keyword arguments. With `epoch_fn=None` and
`selection_start_epoch=1`, Model 0 still uses `train_epoch`, validation NDCG@10
selection, and the same early-stopping rule.

- `epoch_fn(epoch, model, loader, optimizer, device, grad_clip_norm)` replaces
  one training epoch and returns `(finite_loss, positive_integer_n_targets)`.
  Epoch numbers are absolute and 1-based, including after resume. The hook owns
  the forward/backward/update loop; the trainer owns the optimizer, scheduler,
  validation, checkpoint files and early stopping.
- `selection_start_epoch` is the first epoch eligible for **both** best-model
  selection and early stopping. It must be an integer from 1 through the total
  epoch budget. Earlier validation scores are still logged, but do not set the
  best score, stopping reference or patience counter. Stage 1 produces `last.pt`
  for recovery, with no `best.pt` until the first eligible epoch completes.

## Connection to the distillation module in PR #7

After the module is merged, the adapter can be passed without changing it:

```python
from dataclasses import asdict
from src.models.distill import make_epoch_fn
from src.training.trainer import fit

# targets and schedule come from the Model 1 caller, using training-only data.
metadata = {
    **metadata,
    "distill_schedule": asdict(schedule),
    "profile_targets_sha256": targets_file_sha256,
    "distill_source_sha256": distill_source_sha256,
}
fit(model, train_loader, validate, settings, output_dir,
    metadata=metadata,
    epoch_fn=make_epoch_fn(targets, schedule),
    selection_start_epoch=schedule.selection_starts_at())
```

The caller must record the actual schedule, target hashes, model/source hashes
and other run provenance in `metadata`. The trainer records the callable's
`module.qualname` (or callable class name) and the selection start in the
`trainer_options` section of `run_config.json` and `last.pt`. The existing
`training` / `settings` dictionaries still contain only `TrainingSettings`
fields, so Part F can reconstruct that dataclass as before. A name alone cannot
identify captured parameters, a partial's bound arguments or target contents. The callable itself
is not pickled: on resume the caller must supply it again with matching metadata.

Resume rejects changed hook identity or selection start. Older schema-1 Model 0
checkpoints without `trainer_options` are treated as `None` / `1`; they can
still resume with the defaults. For custom hooks, compare `trainer_options`
along with the caller's metadata when handing off runs for evaluation.

This change supplies the trainer interface only. It does not select alpha,
stage-1 duration, profile encoder, or embedding fusion, and does not run Model 1
on the real dataset. Test with the repository's pinned environment:

```sh
python -m unittest discover -s tests -p 'test_training*.py' -v
```
