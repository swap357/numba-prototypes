@echo off
setlocal enabledelayedexpansion

REM Install the package using pip
python -m pip install . -vv --no-deps --no-build-isolation
if errorlevel 1 exit 1

REM Copy tutorial files to a shared location for easy access
mkdir "%PREFIX%\share\sealir-tutorials" 2>nul
copy *.py "%PREFIX%\share\sealir-tutorials\"
copy *.ipynb "%PREFIX%\share\sealir-tutorials\"
xcopy tests "%PREFIX%\share\sealir-tutorials\tests\" /E /I
copy Makefile "%PREFIX%\share\sealir-tutorials\"
copy README.md "%PREFIX%\share\sealir-tutorials\"
copy jupytext.toml "%PREFIX%\share\sealir-tutorials\"
copy pyproject.toml "%PREFIX%\share\sealir-tutorials\"

exit 0