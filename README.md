# CUDA-examples

## Configure on Visual Studio 2019
Solution Congiguration
Release x64

### CUDA
Project --> Build Dependencies --> Build Customizations
select CUDA 10.1

Project --> Properties--> Linker --> General --> Additional Library Directories
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib

Project --> Properties--> Linker --> Input --> Additional Dependencies
cudart_static.lib

