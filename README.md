# Convolutional Neural Network (C++ / OpenCV)

Small end-to-end CNN prototype written in modern C++ that loads image datasets with OpenCV and trains a hand-rolled model. The project is geared toward Windows and uses Conan to fetch dependencies and CMake/Visual Studio to build.

## Dataset Layout
- Default dataset location: `E:\AI_DATA\TrainAndValidateData`. Set `DATASET_PATH` to override (e.g., `set DATASET_PATH=D:\data\cnn`).
- Expected structure under the dataset folder:
  - One subfolder per class with `.jpg` images. Class names the code maps today: `Bikes`, `Buffalo`, `Cars`, `Elephant`, `Motorcycles`, `Planes`, `Rhino`, `Ships`, `Trains`, `Zebra`.
- Images are resized/processed on load; grayscale vs. RGB is controlled in code via `ImageLoader` settings (currently RGB, 120x120).

## Prerequisites (Windows)
- Visual Studio 2022 with the C++ toolset and CMake components.
- Conan 2.x in `PATH` (run `conan --version` to verify). Run once: `conan profile detect --force`.
- Python is not required; OpenCV/Eigen/Boost come from Conan.
- Use a "x64 Native Tools Command Prompt for VS 2022" or "Developer PowerShell for VS 2022" to ensure environment variables are set.

## Quick Start (clean build)
From the repository root:

- Debug (cleans `build/`, installs deps, configures, builds, runs):
  ```
  run_debug.bat
  ```
- Release build/run:
  ```
  run_release.bat
  ```

The scripts:
1) Remove the `build` directory for a clean configuration.
2) `conan install` with `--build=missing` into `build/build/generators`.
3) Configure CMake with the generated toolchain (`build/build/generators/conan_toolchain.cmake`) for VS 2022.
4) Build with `cmake --build` and execute the resulting `cnn.exe`.

Executables live in `build/Debug/cnn.exe` or `build/Release/cnn.exe`. If you only want to build (not run), comment out the last line of the chosen `.bat`.

## Running
- Ensure `DATASET_PATH` points at your dataset before invoking the script, otherwise the default path above is used.
- The program performs an 80/20 train/validation split and will exit if no images are found.
- If you run the `.exe` manually, do it from the `build` directory so the Conan-generated DLLs are found.

## Troubleshooting
- **OpenCV DLL load errors**: Make sure you run `cnn.exe` from inside `build` after executing the corresponding `.bat` so the Conan runtime files are on the relative path.
- **Stale configuration**: Rerun `run_debug.bat` or `run_release.bat` to clean and regenerate the build tree.
- **Dataset issues**: Confirm subfolders exist with the class names above and contain `.jpg` files; the loader skips missing/empty folders.
