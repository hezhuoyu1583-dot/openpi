#!/usr/bin/env python
"""
Wrapper script that sets OpenGL environment variables BEFORE any imports,
then runs the main.py simulation. This handles headless/osmesa rendering.
"""
import os
import sys
import pathlib
import logging
from datetime import datetime

# Get the script directory
script_dir = pathlib.Path(__file__).parent.parent.parent

# Add paths to PYTHONPATH before any imports
sys.path.insert(0, str(script_dir / "src"))  # For openpi_client
sys.path.insert(0, str(script_dir / "packages/openpi-client/src"))  # Alternative path
sys.path.insert(0, str(script_dir / "third_party/libero"))

# Set environment variables BEFORE any imports - this is critical!
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["OSMESA_DEPTH_BITS"] = "24"
os.environ["DISPLAY"] = ""
os.environ["PYTHONPATH"] = ":".join(sys.path) + ":" + os.environ.get("PYTHONPATH", "")

# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = script_dir / f"data/libero/simulation_{timestamp}.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Monkey-patch mujoco module initialization to force osmesa before EGL tries to load
from unittest.mock import MagicMock

# Prevent eglcontext from importing before we set up osmesa
sys.modules["mujoco.egl"] = MagicMock()
sys.modules["mujoco.egl.egl_ext"] = MagicMock()
sys.modules["OpenGL.EGL"] = MagicMock()

# Now import and run main
import main
import tyro

if __name__ == "__main__":
    # Convert positional arguments to typed arguments for tyro
    # This allows: python main_osmesa.py --host 127.0.0.1 --port 8000 ...
    # Instead of: python main_osmesa.py --args.host 127.0.0.1 --args.port 8000 ...
    argv = sys.argv[1:]
    tyro_argv = []
    for arg in argv:
        if arg.startswith("--"):
            # Convert --name to --args.name
            tyro_argv.append("--args." + arg[2:])
        else:
            tyro_argv.append(arg)
    sys.argv = [sys.argv[0]] + tyro_argv
    tyro.cli(main.eval_libero)
