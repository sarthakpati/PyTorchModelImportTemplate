CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)

PROJECT( PyTorchModelImportTemplate VERSION 1.0.0 )
SET( CMAKE_CXX_STANDARD 11 )

FIND_PACKAGE(Torch REQUIRED) # deep learning, baby
FIND_PACKAGE( ITK REQUIRED ) # needed for medical image IO
INCLUDE( "${ITK_USE_FILE}" ) # sets up ITK's includes properly

ADD_SUBDIRECTORY( src/cbica_toolkit ) # adds all the nice common functionality from CaPTk 

INCLUDE_DIRECTORIES( ${CACHED_INCLUDE_DIRS} ) # include directories will be found properly

ADD_EXECUTABLE( ${PROJECT_NAME} src/main.cxx )

ADD_DEPENDENCIES( ${PROJECT_NAME} CBICA_Basic )

TARGET_LINK_LIBRARIES( 
  ${PROJECT_NAME} 
  ${TORCH_LIBRARIES} 
  ${ITK_LIBRARIES} 
  CBICA_Basic
)
