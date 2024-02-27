# Audacity OpenVINO module build for Windows :hammer:

Hi! The following is the process that we use when building the Audacity modules for Windows.

## High-Level Overview
Before we get into the specifics, at a high-level we will be doing the following:
* Cloning & building whisper.cpp with OpenVINO support (For transcription audacity module)
* Cloning & building openvino-stable-diffusion-cpp (This is to support audio generation / remix features)
* Cloning & building Audacity 3.4.2 without modifications (just to make sure 'vanilla' build works fine)
* Adding our OpenVINO module src's to the Audacity source tree, and re-building it.

## Dependencies
Here are some of the dependencies that you need to grab. If applicable, I'll also give the cmd's to set up your environment here.
* CMake (https://cmake.org/download/)
* Visual Studio (MS VS 2019 / 2022 Community Edition is fine)
* python3 / pip - Audacity requires conan 2.0+ to be installed, and the recommended way to do that is through pip.  
* OpenVINO - You can use public version from [here](https://github.com/openvinotoolkit/openvino/releases/tag/2023.1.0). Setup your cmd.exe shell environment by running setupvars.bat:  
    ```
    call "C:\path\to\w_openvino_toolkit_windows_xxxx\setupvars.bat"
    ```
* OpenCV - Only a dependency for the openvino-stable-diffusion-cpp samples (to read/write images from disk, display images, etc.). You can find pre-packages Windows releases [here](https://github.com/opencv/opencv/releases). We currently use 4.8.1 with no issues, it's recommended that you use that.
   ```
   set OpenCV_DIR=C:\path\opencv\build
   set Path=%OpenCV_DIR%\x64\vc16\bin;%Path%
   ```
* Libtorch (C++ distribution of pytorch)- This is a dependency for the audio utilities in openvino-stable-diffusion-cpp (like spectrogram-to-wav, wav-to-spectrogram), as well as some of our htdemucs v4 routines (supporting music separation). We are currently using this version: [libtorch-win-shared-with-deps-2.1.1+cpu.zip](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.1.1%2Bcpu.zip). After extracting the package, setup environment like this:
    ```
    set LIBTORCH_ROOTDIR=C:\path\to\libtorch-shared-with-deps-2.1.1+cpu\libtorch
    set Path=%LIBTORCH_ROOTDIR%\lib;%Path%
    ```

## Sub-Component builds
We're now going to build whisper.cpp and openvino-stable-diffusion-cpp. You should have a cmd.exe (not powershell!) shell running, and environment setup for above dependencies. To recap:  

    :: OpenVINO
    call "C:\path\to\w_openvino_toolkit_windows_xxxx\setupvars.bat"

    :: OpenCV
    set OpenCV_DIR=C:\path\to\opencv\build
    set Path=%OpenCV_DIR%\x64\vc16\bin;%Path%

    :: Libtorch
    set LIBTORCH_ROOTDIR=C:\path\to\libtorch-shared-with-deps-2.1.1+cpu\libtorch
    set Path=%LIBTORCH_ROOTDIR%\lib;%Path%

### Whisper.cpp 
```
:: Clone it  & check out specific commit
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
git checkout ec7a6f04f9c32adec2e6b0995b8c728c5bf56f35
cd ..

:: Create build folder
mkdir whisper-build
cd whisper-build

:: Run CMake, specifying that you want to enable OpenVINO support.
:: Note: Replace visual studio version if needed
cmake ..\whisper.cpp -G "Visual Studio 16 2019" -A x64 -DWHISPER_OPENVINO=ON

:: Build it:
cmake --build . --config Release

:: Install built whisper collateral into a local 'installed' directory:
cmake --install . --config Release --prefix .\installed
```
With the build / install complete, the Audacity build will find the built collateral via the WHISPERCPP_ROOTDIR. So you can set it like this:
```
set WHISPERCPP_ROOTDIR=C:\path\to\whisper-build\installed
set Path=%WHISPERCPP_ROOTDIR%\bin;%Path%
```
(I'll remind you later about this though)

### OpenVINO Stable-Diffusion CPP
```
:: clone it & check out v0.1 tag
git clone https://github.com/intel/stablediffusion-pipelines-cpp.git
cd stablediffusion-pipelines-cpp
git checkout v0.1
cd ..


:: create build folder
mkdir stablediffusion-pipelines-cpp-build
cd stablediffusion-pipelines-cpp-build

:: run cmake
cmake ../stablediffusion-pipelines-cpp

:: Build it:
cmake --build . --config Release

:: Install built collateral into a local 'installed' directory:
cmake --install . --config Release --prefix ./installed

```

With the build / install complete, the Audacity build will find the built collateral via the CPP_STABLE_DIFFUSION_OV_ROOTDIR. So you can set it like this:
```
set CPP_STABLE_DIFFUSION_OV_ROOTDIR=C:\path\to\stablediffusion-pipelines-cpp-build\installed
set Path=%CPP_STABLE_DIFFUSION_OV_ROOTDIR%\bin;%Path%
```
(I'll remind you later about this though)

## Audacity 

Okay, moving on to actually building Audacity. Just a reminder, we're first going to just build Audacity without any modifications. Once that is done, we'll copy our openvino-module into the Audacity src tree, and built that.

### Audacity initial (vanilla) build
```
:: For Audacity 3.4.0+, conan 2.0+ is required. Install it like this:
pip install conan

:: clone Audacity
git clone https://github.com/audacity/audacity.git

:: Check out Audacity-3.4.2 tag, as this is the specific version that
:: our modules are compatible with.
cd audacity
git checkout Audacity-3.4.2
cd ..

mkdir audacity-build
cd audacity-build

:: Run cmake (grab a coffee & a snack... this takes a while)
cmake ..\audacity -G "Visual Studio 16 2019" -A x64 -DAUDACITY_BUILD_LEVEL=2

:: build it 
cmake --build . --config Release
```

When this is done, you can run Audacity like this (from audacity-build directory):
```
Release\Audacity.exe
```

### Audacity OpenVINO module build

Now we'll run through the steps to actually build the OpenVINO-based Audacity module.

First, clone this repo. This is of course where the actual Audacity module code lives.
```
:: clone it
git clone https://github.com/intel/openvino-plugins-ai-audacity.git

cd openvino-plugins-ai-audacity

:: Check out appropriate branch or tag 
git checkout v3.4.2-R1
```

We need to copy the ```mod-openvino``` folder into the Audacity source tree.
i.e. Copy ```openvino-plugins-ai-audacity\mod-openvino``` folder to ```audacity\modules```.

We now need to edit ```audacity\modules\CMakeLists.txt``` to add mod-openvino as a build target. You just need to add a ```add_subdirectory(mod-openvino)``` someplace in the file. For example:

```
...
foreach( MODULE ${MODULES} )
   add_subdirectory("${MODULE}")
endforeach()

#YOU CAN ADD IT HERE
add_subdirectory(mod-openvino)

if( NOT CMAKE_SYSTEM_NAME MATCHES "Darwin" )
   if( NOT "${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
      install( DIRECTORY "${_DEST}/modules"
               DESTINATION "${_PKGLIB}" )
   endif()
endif()
...
```

Okay, now we're going to (finally) build the module. Here's a recap of the environment variables that you should have set:

```
:: OpenVINO
call "C:\path\to\w_openvino_toolkit_windows_xxxx\setupvars.bat"

:: Libtorch
set LIBTORCH_ROOTDIR=C:\path\to\libtorch-shared-with-deps-2.0.1+cpu\libtorch
set Path=%LIBTORCH_ROOTDIR%\lib;%Path%

:: Whisper.cpp 
set WHISPERCPP_ROOTDIR=C:\path\to\whisper-build\installed
set Path=%WHISPERCPP_ROOTDIR%\bin;%Path%

:: C++ Stable Diffusion Pipelines using OpenVINO(TM)
set CPP_STABLE_DIFFUSION_OV_ROOTDIR=C:\path\to\stablediffusion-pipelines-cpp-build\installed
set Path=%CPP_STABLE_DIFFUSION_OV_ROOTDIR%\bin;%Path%
```

Okay, on to the build:  
```
:: cd back to the same Audacity folder you used to build Audacity before
cd audacity-build

:: and build the new target, mod-openvino.
:: (Note: CMake will automatically re-run since you modified CMakeLists.txt)
cmake --build . --config Release --target mod-openvino
```

If it all builds correctly, you should see mod-openvino.dll sitting in audacity-build\bin\Release\modules.

You can go ahead and run audacity-build\bin\Release\Audacity.exe

Once Audacity is open, you need to go to ```Edit -> Preferences```. And on the left side you'll see a ```Modules``` tab, click that. And here you (hopefully) see mod-openvino entry set to ```New```. You need to change it to ```Enabled```, as shown in the following picture.  

![](preferences_enabled.png)

Once you change to ```Enabled```, close Audacity and re-open it. When it comes back up, you should now see the OpenVINO modules listed.

## Installing the OpenVINO models
In order for the OpenVINO effects to work, you need to install the OpenVINO models. 

1. Download [openvino-models.zip](https://github.com/intel/openvino-plugins-ai-audacity/releases/download/v3.4.2-R1/openvino-models.zip).
2. Copy ```openvino-models``` folder from the zip file, such that it is placed in the same folder as ```Audacity.exe``` (e.g. ```audacity-build\bin\Release\```).


# Need Help? :raising_hand_man:
For any questions about this build procedure, feel free to submit an issue [here](https://github.com/intel/openvino-plugins-ai-audacity/issues)

