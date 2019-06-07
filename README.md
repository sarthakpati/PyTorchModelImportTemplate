# PyTorchModelImportTemplate
An example shoowing how to import a PyTorch model and use it using C++

All code has been adopted from https://pytorch.org/tutorials/advanced/cpp_export.html

## Depedencies

* Download LibTorch binaries:
  * https://pytorch.org/get-started/locally/

* Build from Source
  * Python 3.6 from Anaconda - needed for PyTorch installation

  * CMake 3.12 (3.13 and 3.14 have a configuration change with superbuild that makes things a bit messy) - needed for PyTorch installation

  * PyTorch:

    * Clone https://github.com/pytorch/pytorch; use a tag, preferably
    * Update submodules
    * Use CMake to configure and generate binaries.
      * [WINDOWS USERS] use Visual Studio 2017 x64 as the generator from the GUI and add the variable **CMAKE_DEBUG_POSTFIX** as a **STRING** of value **d** (this makes linking against debug libs easier to maintain)
    * Enable the following the flags **CAFFE2_STATIC_LINK_CUDA** and **TORCH_STATIC** and disable **BUILD_SHARED_LIBS** for static compiles 
    * An analogous command set in PowerShell or Bash is as follows:
  
```
git clone https://github.com/pytorch/pytorch.git # use a tag, preferably
cd pytorch
git submodule update --init --recursive # updates all submodules 
mkdir bin; cd bin
cmake -DCMAKE_INSTALL_PREFIX=./install .. # this will use the default generator for the system
```

  * Run the **INSTALL** target (either build that on VS or run `make install/strip` on bash)

* ITK - for image I/O; can be replaced with something different, if needed

  * Clone https://github.com/InsightSoftwareConsortium/ITK
  * Checkout tag 4.13.2 - this is because the I/O dependencies have not yet been checked for 5.x
  * Use CMake to configure and generate binaries 
  * An analogous command set in PowerShell or Bash is as follows:
  
```
git clone https://github.com/InsightSoftwareConsortium/ITK.git # use a tag, preferably
git checkout tags/v4.13.2
cd ITK
mkdir bin; cd bin
cmake -DCMAKE_INSTALL_PREFIX=./install .. # this will use the default generator for the system
```