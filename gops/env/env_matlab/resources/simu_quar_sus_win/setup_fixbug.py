import setuptools, pybind11.setup_helpers, pathlib

PROJECT_ROOT = pathlib.Path(".")
assert pathlib.Path(__file__).parent.absolute() == PROJECT_ROOT.absolute(), "Must compile in project folder!"

# Import extensions from the original setup.py by extracting only the extensions part
def get_extensions_from_setup():
    """Extract extensions configuration from setup.py without executing it"""
    with open("setup.py", "r", encoding="utf-8") as f:
        setup_content = f.read()
    
    # Find the start and end of extensions definition
    import re
    start_pattern = r'extensions = \['
    start_match = re.search(start_pattern, setup_content)
    if not start_match:
        raise ValueError("Could not find extensions definition in setup.py")
    
    start_pos = start_match.start()
    # Find the matching closing bracket
    bracket_count = 0
    end_pos = -1
    for i, char in enumerate(setup_content[start_pos:], start_pos):
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0:
                end_pos = i + 1
                break
    
    if end_pos == -1:
        raise ValueError("Could not find end of extensions definition")
    
    # Extract and execute the extensions definition
    extensions_code = setup_content[start_pos:end_pos]
    local_vars = {
        'pybind11': pybind11,
        'pathlib': pathlib,
        'PROJECT_ROOT': PROJECT_ROOT,
        'str': str
    }
    exec(extensions_code, local_vars)
    return local_vars['extensions']

extensions = get_extensions_from_setup()

# extensions = [
#   pybind11.setup_helpers.Pybind11Extension(
#     name="quarter_sus_win",
#     sources=[
#       str(PROJECT_ROOT / "model" / "quarter_sus_win.cpp"),
#       str(PROJECT_ROOT / "model" / "quarter_sus_win_data.cpp"),
#       str(PROJECT_ROOT / "model" / "rtGetNaN.cpp"),
#       str(PROJECT_ROOT / "model" / "rt_nonfinite.cpp"),
      
#       str(PROJECT_ROOT / "module.cc"),
#     ],
#     include_dirs=[
#       str(PROJECT_ROOT / "include"),
#       str(PROJECT_ROOT / "model"),
#     ],
#     define_macros=[
#       ("FMT_HEADER_ONLY", None),
#       ("PORTABLE_WORDSIZES", None),
#       ("_CRT_SECURE_NO_WARNINGS", None),
#       ("SLXPY_EXTENSION_NAME", "quarter_sus_win"),
#       ("SLXPY_EXTENSION_VERSION", "15.2"),
#       ("SLXPY_EXTENSION_AUTHOR", "hjzsj"),
#     ],
#     cxx_std=17,
#   )
# ]

from setuptools.command.build_ext import build_ext

class custom_build_ext(build_ext):
  user_options = build_ext.user_options + [
    ('no-stub', 'S', 'Skip stub generation')
  ]

  def initialize_options(self):
    super().initialize_options()
    self.no_stub = None

  def finalize_options(self):
    super().finalize_options()

  def build_extensions(self):
    super().build_extensions()

    if not self.no_stub:
      for ext in self.extensions:
        if isinstance(ext, pybind11.setup_helpers.Pybind11Extension):
          self._generate_stub(ext)
        else:
          print(f"Skipping stub generation for extension {ext.name}")

  def _generate_stub(self, ext):
    try:
      from pybind11_stubgen import main as stubgen
    except:
      import warnings
      warnings.warn("pybind11-stubgen not found, stubs will not be generated")
      return

    ext_name: str = ext.name
    import sys
    build_lib = pathlib.Path(self.build_lib).absolute()
    sys.path.insert(0, str(build_lib))

    stubgen([
      "-o", str(build_lib),
      "--root-suffix", "",
      ext_name
    ])
    stub_dir = build_lib / ext_name
    for stub_file in stub_dir.glob("*.pyi"):
      self._postprocess_stub(stub_file)

  def _postprocess_stub(self, stub_file: pathlib.Path):
    stub = stub_file.read_text()
    stub = stub.replace("numpy.dtype[void]", "numpy.dtype[numpy.void]")
    stub_file.write_text(stub)

setuptools.setup(ext_modules=extensions, cmdclass={ "build_ext": custom_build_ext })
