
import os
import sys

sys.path += [os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "build")]

if os.name == "nt" and sys.version_info[1] >= 8:
    for path in os.environ["PATH"].split(";"):
        if path != "" and os.path.exists(os.path.abspath(path)):
            os.add_dll_directory(os.path.abspath(path))
