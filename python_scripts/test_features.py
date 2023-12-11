from math import sqrt
import features
import numpy as np



def test_line_endpoints_distance():
    line1 = np.array([0,0,1,0])
    line2 = np.array([1,1,1,2])
    expected = np.zeros((2,2))
    expected[0,0] = sqrt(2)
    expected[0,1] = sqrt(5)
    expected[1,0] = 1
    expected[1,1] = 2
    assert np.allclose(features.line_endpoints_distance(line1, line2), expected)

def test_line_proximity():
    line1 = np.array([0,0,1,0])
    line2 = np.array([1,1,1,2])
    assert features.line_proximity(line1, line2, 1.5)
    assert not features.line_proximity(line1, line2, 0.5)
