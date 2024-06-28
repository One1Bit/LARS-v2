# LARS-v2

LARS v2 is an advanced application for music source separation and drums demixing. It builds upon the foundation of LARS v1, enhancing its capabilities to handle complete musical tracks.

![Screenshot 2024-06-28 at 10 41 36](https://github.com/One1Bit/LARS-v2/assets/45692058/f53014fd-e7c2-4e2c-a6e7-ce220ba1ceca)
![Screenshot 2024-06-28 at 10 38 10](https://github.com/One1Bit/LARS-v2/assets/45692058/864025ae-4801-4231-852e-64f9e9bd6e73)

## Description

The implementation of LARS v2 involves integrating pre-trained models such as HTDemucs4 FT Drums and LarsNet (from LARS v1) into the C++ codebase. 

LARS v2 can operate in two modes: the same drum-focused mode (separate a stereo drum track into five audio stems: **kicks**, **snare**, **toms**, **hi-hat**, and **cymbals**.) as LARS v1 
and a full-featured plugin mode that can separate complete musical tracks, because of that we added 6th stem **drums**.

## Requirements
* [CMake](https://cmake.org) 
* [Libtorch](https://pytorch.org/get-started/locally/)
* [Juce](https://juce.com)

## How to run LARS

* Clone this repo
* In the `CMakeLists.txt` file, modify the following lines by typing in the path to your LibTorch and JUCE folders:
  * `add_subdirectory(/Path/../JUCE ./JUCE)`
* On your terminal, go to the project folder and run:
```console
cmake -B build .
```
* Or build the project with VS code.

## License
[LarsNet](https://github.com/polimi-ispl/larsnet), whose weights are distributed under a CC BY-NC 4.0 license.
[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training?tab=readme-ov-file)
