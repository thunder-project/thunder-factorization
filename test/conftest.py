import pytest
import station

def pytest_addoption(parser):
    parser.addoption("--engine", action="store", default="spark",
                     help="engine to run tests with")

@pytest.fixture(scope='module')
def eng(request):
    station.start(spark=True)
    return station.engine()
