import rpd.cube_state as cube_state
import numpy as np

SOLVED_STATE_ARRAY = np.array([
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    [
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ],
    [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ],
    [
        [4, 4, 4],
        [4, 4, 4],
        [4, 4, 4],
    ],
    [
        [5, 5, 5],
        [5, 5, 5],
        [5, 5, 5],
    ],
])

def test_cube_to_solver_string():
    state = cube_state.CubeState(SOLVED_STATE_ARRAY)
    assert state.to_solver_string() == "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

def test_cube_solved():
    state = cube_state.CubeState(SOLVED_STATE_ARRAY)
    assert state.is_solved(), "The cube should be solved"

def test_cube_rotate_face():
    cube_state.CubeState(SOLVED_STATE_ARRAY)
    assert 1 == 0