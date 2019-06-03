# PyTorchModelImportTemplate
An example shoowing how to import a PyTorch model and use it using C++

All code has been adopted from https://pytorch.org/tutorials/advanced/cpp_export.html

## Depedencies
* Python 3.6 from Anaconda - needed for PyTorch installation

* PyTorch:

  * Clone https://github.com/pytorch/pytorch; use a tag, preferably
  * Update submodules
  * Use CMake to configure and generate binaries 
  * An analogous command set in PowerShell or Bash is as follows:
  
```
git clone https://github.com/pytorch/pytorch.git # use a tag, preferably
cd pytorch
git submodule update --init --recursive # updates all submodules 
mkdir bin; cd bin
cmake -DCMAKE_INSTALL_PREFIX=./install .. # this will use the default generator for the system
```

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