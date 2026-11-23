from io import StringIO

import numpy as np
import pytest


@pytest.fixture
def concrete_csv():
    header = ("Cement,Blast Furnace Slag,Fly Ash,Water,Superplasticizer,Coarse Aggregate,"
              "Fine Aggregate,Age,Concrete compressive strength")
    index = np.arange(1030)
    recipe = index % 103
    ingredients = np.column_stack([recipe * k for k in range(1, 8)])
    table = np.column_stack((ingredients, 1 + index // 103, 1 + index))
    stream = StringIO()
    np.savetxt(stream, table, fmt="%d", delimiter=",", header=header, comments="")
    return stream.getvalue().encode()
