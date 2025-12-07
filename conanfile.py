import os

from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain, CMake, cmake_layout
from conan.tools.build import check_min_cppstd, check_max_cppstd
from conan.errors import ConanInvalidConfiguration
from conan.tools.files import copy
from conan.tools.gnu import AutotoolsToolchain, Autotools, PkgConfigDeps

class ConvolutionalNeuralNetwork(ConanFile):
    name = "cnn"
    version = "1.0"
    
    # optional metadata
    license = ""
    author = "Rajiv Sithiravel rajiv.sithiravel@gmail.com"
    url = ""
    description = ""
    
    # binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False], "optimized": [1, 2, 3]}
    default_options = {"shared": False, "fPIC": True, "optimized": 1}
    
    # sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "cnn/*"
        
    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC
            
    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")
    
    def validate(self):
        # OS support check
        supported_os = {
            "Windows",
            "Linux",
            "iOS",
            "watchOS",
            "tvOS",
            "visionOS",
            "Macos",
            "Android",
            "FreeBSD",
            "SunOS",
            "AIX",
            "Arduino",
            "Emscripten",
            "Neutrino",
            "baremetal",
            "VxWorks",
        }
        if str(self.settings.os) not in supported_os:
            raise ConanInvalidConfiguration(f"No support for the operating system: {self.settings.os}")

        # C++ standard checks (align with CMake's C++20)
        check_min_cppstd(self, "20")
        check_max_cppstd(self, "23")
    
    # Needed libraries
    def requirements(self):
        self.requires("boost/1.85.0")
        self.requires("eigen/3.4.0")
        self.requires("opencv/4.9.0")
        
    def build_requirements(self):
        self.tool_requires("cmake/3.30.1")
    
    def layout(self):
        cmake_layout(self)
          
    def generate(self):
        if self.settings.os == "Windows" or self.settings.os == "Linux" :
            tc = CMakeToolchain(self)
            tc.generate()
            deps = CMakeDeps(self)
            deps.generate()
        else:
            tc = AutotoolsToolchain(self)
            tc.generate()
            deps = PkgConfigDeps(self)
            deps.generate()
            
    def build(self):
        if self.settings.os == "Windows" or self.settings.os == "Linux" :
            cmake = CMake(self)
            cmake.configure()
            cmake.build()
        else:
            autotools = Autotools(self)
            autotools.autoreconf()
            autotools.configure()
            autotools.make()

    def package(self):
        cmake = CMake(self)
        cmake.install()
    
