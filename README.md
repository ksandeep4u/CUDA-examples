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

### OpenCV
Project --> Properties--> C/C++ --> General --> Additional Include Directories
$root-folder$\lib\opencv3.3.0\include

Project --> Properties--> Linker --> General --> Additional Library Directories
$root-folder$\lib\opencv3.3.0\lib64

Project --> Properties--> Linker --> Input --> Additional Dependencies
opencv_highgui330.lib
opencv_imgproc330.lib
opencv_core330.lib
opencv_videoio330.lib
opencv_cudaimgproc330.lib
opencv_cudafeatures2d330.lib
opencv_cudaarithm330.lib
opencv_cudafilters330.lib
