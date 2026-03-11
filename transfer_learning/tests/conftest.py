import pytest


@pytest.fixture(params=["lamar", "hoh"])
def watershed(request):
    """Parametrized fixture providing watershed names."""
    return request.param
