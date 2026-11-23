
from minesweeper.manager import GameManager
from minesweeper.recognition.detection import GridDetection


def box(x, y, width=288, height=288):
    return GridDetection(x, y, width, height)


def test_creates_multiple_sessions():
    manager = GameManager()

    manager.update([
        box(100, 100),
        box(600, 100),
    ])

    assert len(manager.sessions) == 2
    assert manager.sessions[0].number == 1
    assert manager.sessions[1].number == 2
    assert manager.sessions[0].id != manager.sessions[1].id


def test_small_movement_preserves_identity():
    manager = GameManager()

    manager.update([box(100, 100)])
    original_id = manager.sessions[0].id

    manager.update([box(125, 115)])

    assert len(manager.sessions) == 1
    assert manager.sessions[0].id == original_id
    assert manager.sessions[0].bbox.x == 125
    assert manager.sessions[0].bbox.y == 115


def test_detection_order_does_not_change_identity():
    manager = GameManager()

    manager.update([
        box(100, 100),
        box(600, 100),
    ])

    first_id = manager.sessions[0].id
    second_id = manager.sessions[1].id

    # Reverse detector output order.
    manager.update([
        box(610, 110),
        box(110, 110),
    ])

    assert manager.sessions[0].id == first_id
    assert manager.sessions[1].id == second_id


def test_temporarily_missing_game_is_retained():
    manager = GameManager(max_missed_scans=2)

    manager.update([box(100, 100)])
    original_id = manager.sessions[0].id

    manager.update([])

    assert len(manager.sessions) == 1
    assert manager.sessions[0].missed_scans == 1

    # Game reappears.
    manager.update([box(110, 110)])

    assert len(manager.sessions) == 1
    assert manager.sessions[0].id == original_id
    assert manager.sessions[0].missed_scans == 0


def test_missing_game_is_eventually_retired():
    manager = GameManager(max_missed_scans=2)

    manager.update([box(100, 100)])
    manager.update([])
    manager.update([])

    assert manager.sessions == []


def test_new_game_does_not_reuse_game_number():
    manager = GameManager(max_missed_scans=1)

    manager.update([box(100, 100)])
    manager.update([])

    manager.update([box(600, 100)])

    assert len(manager.sessions) == 1
    assert manager.sessions[0].number == 2


def test_resize_preserves_identity_when_overlap_is_sufficient():
    manager = GameManager()

    manager.update([box(100, 100, 288, 288)])
    original_id = manager.sessions[0].id

    manager.update([box(100, 100, 350, 350)])

    assert len(manager.sessions) == 1
    assert manager.sessions[0].id == original_id
    assert manager.sessions[0].bbox.width == 350
