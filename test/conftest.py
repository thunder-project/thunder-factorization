import pytest
import station

def pytest_addoption(parser):
    parser.addoption("--engine", action="store", default="local", help="engine to use")

@pytest.fixture(scope='module')
def eng(request):
    engine = request.config.getoption("--engine")
    if engine == 'local':
        return None
    if engine == 'spark':
        station.start(spark=True)
        return station.engine()
