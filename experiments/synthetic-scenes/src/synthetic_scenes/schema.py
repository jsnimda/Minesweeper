"""Public input models and lightweight output types for the active renderer.

The renderer validates dictionaries but continues to use their original values:
model defaults describe omitted options without inserting them into requests.
"""
from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, ValidationInfo, field_validator, model_validator


Seed = StrictInt | StrictFloat | StrictStr | bytes | bytearray | None
SceneMode = Literal['desktop', 'game', 'grid']
GameStyle = Literal['classic', 'windows7', 'microsoft', 'google']
ApplicationType = Literal[
    'browser', 'file_manager', 'spreadsheet', 'document_editor', 'presentation',
    'code_editor', 'terminal', 'calculator', 'settings', 'sudoku', 'other_puzzle',
    'media_viewer',
]


class SceneModel(BaseModel):
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)


class ScreenSpec(SceneModel):
    # Screenshot dimensions are physical pixels. Only unit scale is supported.
    width: Annotated[StrictInt, Field(ge=320)]
    height: Annotated[StrictInt, Field(ge=240)]
    scale: Literal[1, 1.0] = 1.0


class DesktopSpec(SceneModel):
    shell: Literal['taskbar', 'dock', 'top_panel']
    style: Literal['modern', 'aero', 'classic']
    color_scheme: Literal['default', 'light', 'dark']
    accent_color: StrictStr | None = None
    taskbar_layout: Literal['left', 'centered', ''] | None = None
    clock_time: StrictStr = '12:34'
    clock_date: StrictStr = '11/21/2026'

    @field_validator('accent_color')
    @classmethod
    def check_accent_color(cls, value):
        if value:
            digits = value.lstrip('#')
            try:
                if len(digits) < 6:
                    raise ValueError
                int(digits[:6], 16)
            except ValueError:
                raise ValueError('accent_color must start with six hexadecimal digits') from None
        return value


class WallpaperSpec(SceneModel):
    category: StrictStr
    seed: Seed
    subtype: Literal['solid', 'linear', 'radial', 'multi_gradient', ''] | None = None
    placement: Literal['cover', 'fit', 'repeat'] = 'cover'
    background_color: StrictStr | tuple[int, int, int] | tuple[int, int, int, int] | StrictInt | None = '#202020'
    tile_size: tuple[Annotated[StrictInt, Field(gt=0)], Annotated[StrictInt, Field(gt=0)]] | None = None

    @field_validator('category')
    @classmethod
    def check_category(cls, value: str, info: ValidationInfo) -> str:
        assets = (info.context or {}).get('wallpaper_assets')
        categories = {'artwork', 'photographs'} if assets is None else set(assets)
        if value != 'procedural' and value not in categories:
            raise ValueError(f'Unknown wallpaper category: {value}')
        return value


class GameAppearance(SceneModel):
    style: GameStyle = 'classic'
    theme: StrictStr = 'default'
    cell_size: Annotated[StrictInt, Field(ge=8)] = 32
    font_family: StrictStr = 'Arial'
    font_scale: Annotated[StrictFloat, Field(ge=.3, le=.9)] = .72
    bold: StrictBool = True
    digit_palette: tuple[StrictStr, StrictStr, StrictStr, StrictStr, StrictStr, StrictStr, StrictStr, StrictStr] | None = None
    # The renderer ignores boundary_ratio for non-Google styles.
    boundary_ratio: Annotated[StrictFloat, Field(ge=0, le=.25)] = .08
    mine_variant: Literal['classic', 'flower'] = 'classic'

    @field_validator('digit_palette')
    @classmethod
    def check_digit_palette(cls, value):
        if value is not None:
            for color in value:
                try:
                    channels = bytes.fromhex(color.lstrip('#'))
                except ValueError:
                    raise ValueError('digit palette colors must be hexadecimal') from None
                if len(channels) not in (3, 4):
                    raise ValueError('digit palette colors must have three or four channels')
        return value

    @model_validator(mode='after')
    def check_style_options(self):
        themes = {
            'classic': {'default', 'blue', 'green', 'warm'},
            'windows7': {'default', 'blue', 'green'},
            'microsoft': {'default', 'blue', 'warm'},
            'google': {'default', 'blue', 'warm'},
        }
        if self.theme not in themes[self.style]:
            raise ValueError(f'Unknown theme {self.theme!r} for {self.style}')
        if self.mine_variant == 'flower' and self.style != 'windows7':
            raise ValueError('Flower variant is only supported by windows7')
        return self


class GameContent(SceneModel):
    rows: Annotated[StrictInt, Field(gt=0)]
    columns: Annotated[StrictInt, Field(gt=0)]
    mine_count: Annotated[StrictInt, Field(ge=0)]
    status: Literal['new', 'active', 'won', 'lost']
    progress: Annotated[StrictFloat, Field(ge=0, le=1)]
    seed: Seed
    appearance: GameAppearance
    elapsed_seconds: Annotated[StrictInt, Field(ge=0)] = 0

    @model_validator(mode='after')
    def check_game(self):
        if self.mine_count >= self.rows * self.columns:
            raise ValueError('mine_count must be less than the number of cells')
        if self.status == 'active' and not 0 < self.progress < 1:
            raise ValueError('active games require progress strictly between 0 and 1')
        if self.status == 'lost' and self.mine_count == 0:
            raise ValueError('a lost game requires at least one mine')
        if self.status == 'new' and self.elapsed_seconds != 0:
            raise ValueError('new games must have zero elapsed seconds')
        return self


class MinesweeperWindow(SceneModel):
    type: Literal['minesweeper']
    content: GameContent


class ApplicationWindow(SceneModel):
    type: ApplicationType
    seed: Seed


WindowSpec = Annotated[MinesweeperWindow | ApplicationWindow, Field(discriminator='type')]


class SceneSpec(SceneModel):
    mode: SceneMode
    seed: Seed
    windows: list[WindowSpec]
    # Omitted uses the scene seed; explicit None requests a random seed.
    layout_seed: Seed = None
    screen: ScreenSpec | None = None
    desktop: DesktopSpec | None = None
    wallpaper: WallpaperSpec | None = None
    # Dataset sampling records this category in the request for reproducibility.
    cleanliness: StrictStr | None = None

    @model_validator(mode='after')
    def check_mode(self):
        if self.mode == 'desktop':
            if self.screen is None or self.desktop is None or self.wallpaper is None:
                raise ValueError('desktop scenes require screen, desktop, and wallpaper')
        elif len(self.windows) != 1 or not isinstance(self.windows[0], MinesweeperWindow):
            raise ValueError('game and grid scenes require exactly one Minesweeper window')
        return self


Box = tuple[float, float, float, float]


class WindowLayout(TypedDict, total=False):
    index: int
    position: tuple[int, int]
    window_size: tuple[int, int]
    box: Box
    cell_size: int
    content_size: tuple[int, int]


class ResolvedScene(TypedDict, total=False):
    """Requested fields plus optional placement and wallpaper resolution."""
    mode: SceneMode
    seed: Seed
    windows: list[dict]
    screen: dict
    desktop: dict
    wallpaper: dict
    layout_seed: Seed
    cleanliness: str | None
    layout: list[WindowLayout]
    resolved_wallpaper: dict


class BoardAnnotation(TypedDict):
    """Coordinates are in the final image; boxes use half-open xyxy bounds."""
    boxes: dict[str, Box]
    rows: int
    columns: int
    style: GameStyle
    renderer: str
    source: str
    theme: str
    style_parameters: dict
    generation_mode: str
    target_number: int | None
    target_position: tuple[int, int] | None
    board_seed: Seed
    status: str
    board_id: str
    game_state_id: str
    coordinate_space: NotRequired[str]
    grid_origin: NotRequired[tuple[float, float]]
    window_box: NotRequired[Box]
    x_edges: NotRequired[list[float]]
    y_edges: NotRequired[list[float]]
    cell_boxes: NotRequired[list[list[Box]]]
    cell_states: NotRequired[list[list[int]]]
    opened_surfaces: NotRequired[list[list[bool]]]
    mine_positions: NotRequired[list[tuple[int, int]]]


class SceneAnnotations(TypedDict):
    source: Literal['synthetic']
    mode: SceneMode
    seed: Seed
    size: tuple[int, int]
    boards: list[BoardAnnotation]
