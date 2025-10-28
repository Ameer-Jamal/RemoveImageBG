from removebg_app.state import HistoryManager, ImageState


def test_history_push_undo_redo():
    history = HistoryManager()
    base = ImageState(hue_shift=0)
    history.push(base)
    assert history.current().hue_shift == 0
    assert not history.can_undo
    assert not history.can_redo

    state_one = ImageState(hue_shift=10)
    history.push(state_one)
    assert history.can_undo
    assert not history.can_redo

    undone = history.undo()
    assert undone is not None
    assert undone.hue_shift == 0
    assert not history.can_undo
    assert history.can_redo

    redone = history.redo()
    assert redone is not None
    assert redone.hue_shift == 10
    assert history.can_undo
    assert not history.can_redo
