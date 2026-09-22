#[==============================================[
FindLibUSB
-----------

Searching libusb-1.0 library and creating imported 
target LibUSB::LibUSB

#]==============================================]

# TODO Append parts for Version compasion and REQUIRED support

if (NOT TARGET LibUSB::LibUSB)
  if (MSVC OR MINGW)
    find_path(LibUSB_INCLUDE_DIR
      NAMES libusb.h
      PATH_SUFFIXES libusb-1.0
    )
    find_library(LibUSB_LIBRARY
      NAMES usb-1.0 libusb-1.0
    )
    if (LibUSB_INCLUDE_DIR AND LibUSB_LIBRARY)
      message(STATUS "libusb-1.0 found using CMake search paths")
      add_library(LibUSB::LibUSB
        UNKNOWN IMPORTED
      )
      set_target_properties(LibUSB::LibUSB PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LibUSB_INCLUDE_DIR}
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION ${LibUSB_LIBRARY}
      )
    else()
      message(FATAL_ERROR "libusb-1.0 was not found; set LibUSB_INCLUDE_DIR and LibUSB_LIBRARY")
    endif()
  else()
    find_package(PkgConfig)
    pkg_check_modules(LibUSB REQUIRED
      libusb-1.0
    )

    if(LibUSB_FOUND)
      message(STATUS "libusb-1.0 found using pkgconfig")

      add_library(LibUSB::LibUSB
        UNKNOWN IMPORTED
      )
      if (DEFINED LibUSB_INCLUDE_DIRS AND NOT LibUSB_INCLUDE_DIRS STREQUAL "")
        set_target_properties(LibUSB::LibUSB PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES ${LibUSB_INCLUDE_DIRS}
        )
      endif()

      if(LibUSB_LIBRARIES)
        find_library(LibUSB_LIBRARY
          NAMES ${LibUSB_LIBRARIES}
          PATHS ${LibUSB_LIBDIR} ${LibUSB_LIBRARY_DIRS}
        )
        if(LibUSB_LIBRARY)
          set_target_properties(LibUSB::LibUSB PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION ${LibUSB_LIBRARY}
          )
        else()
          message(WARNING "Could not found libusb-1.0 library file")
        endif()
      endif()
    endif()
  endif()
endif()
