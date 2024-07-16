
def pytest_addoption(parser):
    parser.addoption("--symm", action="store", default="su2sz")
    parser.addoption("--fd_data", action="store", default="")