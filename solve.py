from enum import IntEnum
import numpy as np
from . import cube_state
from . import color
import kociemba
from typing import TypeAlias
class MoveDirection(IntEnum):
    CLOCKWISE = 1
    COUNTERCLOCKWISE = -1
    DOUBLE = 2

Move: TypeAlias = tuple[color.SquareColor, MoveDirection]

def solve(state: cube_state.CubeState) -> list[Move]:
    """Solves the state of the cube using the Kociemba solver."""
    def __direction_str_to_int(direction: str) -> int:
        if direction == None or len(direction) == 0:
            return 1
        if direction == "'":
            return -1
        if direction == "2":
            return 2
        raise ValueError(f"Invalid modifier {direction} on move")
    solve_str = kociemba.solve(state.to_solver_string())
    solve_steps = [ (color.spatial_symbol_to_color(move[0]), __direction_str_to_int(move[1:])) for move in solve_str.split(" ")]
    return solve_steps
