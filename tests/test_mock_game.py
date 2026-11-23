import pytest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from minesweeper.devtools.control_panel import DevControlPanel
from minesweeper.devtools.mock_game import MockGameWindow


@pytest.fixture
def app():
    existing = QApplication.instance()

    if existing is not None:
        return existing

    return QApplication([])


def test_create_multiple_games(app):
    panel = DevControlPanel()

    panel.create_game()
    panel.create_game()

    try:
        assert len(panel.games) == 2

        assert panel.games[0].game_number == 1
        assert panel.games[1].game_number == 2

        assert panel.games[0].rows == 9
        assert panel.games[0].cols == 9

        assert panel.games[0] is not panel.games[1]

    finally:
        for game in panel.games:
            game.close()
        panel.close()


def test_mock_game_header(app):
    from minesweeper.devtools.mock_game import MockGameWindow

    game = MockGameWindow(game_number=1)

    try:
        assert game.mine_counter.text() == "010"
        assert game.timer_label.text() == "000"
        assert game.reset_button.text() == "🙂"

        assert game.grid.rows == 9
        assert game.grid.cols == 9
    finally:
        game.close()


def test_click_reveals_cell(app):
    game = MockGameWindow(
        game_number=1,
        rows=3,
        cols=3,
        mine_count=1,
    )

    try:
        game.game.mines = {(1, 1)}

        game.show()
        app.processEvents()

        QTest.mouseClick(
            game.grid,
            Qt.MouseButton.LeftButton,
            pos=QPoint(16, 16),
        )

        assert (0, 0) in game.game.revealed
        assert not game.game.game_over

        QTest.mouseClick(
            game.reset_button,
            Qt.MouseButton.LeftButton,
        )

        assert game.game.revealed == set()
        assert not game.game.game_over

    finally:
        game.close()


def test_overlay_visibility(app):
    panel = DevControlPanel()

    try:
        assert not panel.overlay.isVisible()

        panel.toggle_overlay()
        app.processEvents()

        assert panel.overlay.isVisible()
        assert panel.overlay_button.text() == "Hide Overlay"

        panel.toggle_overlay()
        app.processEvents()

        assert not panel.overlay.isVisible()
        assert panel.overlay_button.text() == "Show Overlay"

    finally:
        panel.close()


def test_mock_detector_finds_visible_games(app):
    panel = DevControlPanel()

    try:
        panel.create_game()
        panel.create_game()
        app.processEvents()

        detections = panel.detector.detect(panel.games)

        assert len(detections) == 2
        assert detections[0].width == 9 * 32
        assert detections[0].height == 9 * 32

        # Hidden games should not be detected.
        panel.games[0].hide()
        app.processEvents()

        detections = panel.detector.detect(panel.games)
        assert len(detections) == 1

    finally:
        panel.close()


def test_scan_updates_overlay(app):
    panel = DevControlPanel()

    try:
        panel.create_game()
        app.processEvents()

        panel.scan_games()

        assert len(panel.overlay.annotations) == 1
        assert panel.detection_label.text() == "Detections: 1"

        first_annotation = panel.overlay.annotations[0]
        original_label = first_annotation.label
        original_x = first_annotation.bbox.x
        original_y = first_annotation.bbox.y

        game = panel.games[0]
        game.move(game.x() + 40, game.y() + 30)
        app.processEvents()

        panel.scan_games()

        new_annotation = panel.overlay.annotations[0]

        assert new_annotation.label == original_label
        assert new_annotation.bbox.x != original_x
        assert new_annotation.bbox.y != original_y

    finally:
        panel.close()


def test_missing_game_annotation_is_hidden(app):
    panel = DevControlPanel()

    try:
        panel.create_game()
        app.processEvents()

        panel.scan_games()

        assert len(panel.manager.sessions) == 1
        assert len(panel.overlay.annotations) == 1

        # Hide the game and scan again.
        panel.games[0].hide()
        app.processEvents()

        panel.scan_games()

        # Session is retained during the grace period.
        assert len(panel.manager.sessions) == 1
        assert panel.manager.sessions[0].missed_scans == 1

        # But its stale annotation should disappear.
        assert panel.overlay.annotations == []

    finally:
        panel.close()
