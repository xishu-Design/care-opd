from care_opd.trainer_utils import (
    ProgressReporter,
    TrainingArguments,
    total_update_steps,
)


def test_total_steps_uses_prepared_three_gpu_dataloader_length() -> None:
    arguments = TrainingArguments(
        gradient_accumulation_steps=32,
        num_train_epochs=1,
    )
    assert total_update_steps(4_400, arguments) == 138


def test_total_steps_respects_max_steps() -> None:
    arguments = TrainingArguments(max_steps=17)
    assert total_update_steps(4_400, arguments) == 17


def test_progress_reporter_emits_nohup_friendly_lines(capsys) -> None:
    progress = ProgressReporter(
        stage="sft:clean_sft",
        total_steps=4,
        initial_step=1,
    )
    progress.update(2, loss=1.25, lr=2e-5)
    progress.status("saving_final")
    progress.complete()

    output = capsys.readouterr().out
    assert "stage=sft:clean_sft status=start step=1/4 pct=25.00%" in output
    assert "status=running step=2/4 pct=50.00%" in output
    assert "loss=1.25" in output
    assert "lr=2e-05" in output
    assert "status=saving_final" in output
    assert "status=complete step=4/4 pct=100.00%" in output


def test_progress_reporter_is_silent_when_disabled(capsys) -> None:
    progress = ProgressReporter(
        stage="rank_1",
        total_steps=3,
        enabled=False,
    )
    progress.update(1, loss=2.0)
    progress.complete()
    assert capsys.readouterr().out == ""
