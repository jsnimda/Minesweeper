
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from minesweeper.devtools.game_model import MockGame


class GridWidget(QWidget):
    CELL_SIZE = 32
    changed = Signal()

    NUMBER_COLORS = {
        1: "#2563eb",
        2: "#15803d",
        3: "#dc2626",
        4: "#7c3aed",
        5: "#991b1b",
        6: "#0891b2",
        7: "#111827",
        8: "#64748b",
    }

    def __init__(self, game: MockGame):
        super().__init__()

        self.game = game
        self.rows = game.rows
        self.cols = game.cols

        self.setFixedSize(
            self.cols * self.CELL_SIZE,
            self.rows * self.CELL_SIZE,
        )

    def mousePressEvent(self, event):
        if event.button() not in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            return

        col = int(event.position().x()) // self.CELL_SIZE
        row = int(event.position().y()) // self.CELL_SIZE

        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.game.reveal(row, col)
        else:
            self.game.toggle_flag(row, col)

        self.update()
        self.changed.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setFont(QFont("Arial", 15, QFont.Weight.Bold))

        size = self.CELL_SIZE

        for row in range(self.rows):
            for col in range(self.cols):
                x = col * size
                y = row * size
                pos = (row, col)

                revealed = pos in self.game.revealed
                flagged = pos in self.game.flags
                mine = pos in self.game.mines

                if revealed:
                    background = "#e0e0e0"
                else:
                    background = "#c0c0c0"

                if self.game.game_over and mine and not self.game.won:
                    background = "#fca5a5"

                painter.fillRect(x, y, size, size, QColor(background))

                text = ""
                color = "#222222"

                if flagged:
                    text = "⚑"
                    color = "#dc2626"
                elif mine and self.game.game_over and not self.game.won:
                    text = "✹"
                    color = "#991b1b"
                elif revealed:
                    if mine:
                        text = "✹"
                        color = "#991b1b"
                    else:
                        count = self.game.adjacent_mines(row, col)
                        if count:
                            text = str(count)
                            color = self.NUMBER_COLORS[count]

                if text:
                    painter.setPen(QColor(color))
                    painter.drawText(
                        x, y, size, size,
                        Qt.AlignmentFlag.AlignCenter,
                        text,
                    )

                painter.setPen(QPen(QColor("#777777"), 1))
                painter.drawRect(x, y, size - 1, size - 1)


class MockGameWindow(QWidget):
    def __init__(
        self,
        game_number: int,
        rows: int = 9,
        cols: int = 9,
        mine_count: int = 10,
    ):
        super().__init__()

        self.game_number = game_number
        self.rows = rows
        self.cols = cols
        self.game = MockGame(rows, cols, mine_count)

        self.setWindowTitle(f"Minesweeper Game {game_number}")

        self.mine_counter = QLabel()
        self.timer_label = QLabel("000")
        self.reset_button = QPushButton("🙂")

        for label in (self.mine_counter, self.timer_label):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(
                "background: #202020;"
                "color: #ff4545;"
                "font: bold 20px monospace;"
                "padding: 4px;"
            )
            label.setMinimumWidth(65)

        self.reset_button.setFixedSize(42, 36)
        self.reset_button.clicked.connect(self.reset_game)

        header = QHBoxLayout()
        header.addWidget(self.mine_counter)
        header.addStretch()
        header.addWidget(self.reset_button)
        header.addStretch()
        header.addWidget(self.timer_label)

        self.grid = GridWidget(self.game)
        self.grid.changed.connect(self.on_game_changed)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addWidget(self.grid)

        self.seconds = 0
        self.timer_started = False

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)

        self.update_header()
        self.setFixedSize(self.sizeHint())

    def tick(self):
        self.seconds = min(self.seconds + 1, 999)
        self.update_header()

        if self.seconds == 999:
            self.timer.stop()

    def update_header(self):
        self.mine_counter.setText(
            f"{self.game.remaining_mines:03d}"
        )
        self.timer_label.setText(f"{self.seconds:03d}")

        if self.game.won:
            self.reset_button.setText("😎")
        elif self.game.game_over:
            self.reset_button.setText("☹")
        else:
            self.reset_button.setText("🙂")

    def on_game_changed(self):
        if self.game.revealed and not self.timer_started:
            self.timer_started = True
            self.timer.start()

        if self.game.game_over:
            self.timer.stop()

        self.update_header()

    def reset_game(self):
        self.timer.stop()
        self.seconds = 0
        self.timer_started = False

        self.game.reset()
        self.update_header()
        self.grid.update()
