include $(top_srcdir)/Common.mk




##     .########.##.....##.########
##     .##........##...##.....##...
##     .##.........##.##......##...
##     .######......###.......##...
##     .##.........##.##......##...
##     .##........##...##.....##...
##     .########.##.....##....##...


define _ext_bootstrap
$(eval
$1/configure: | $1
	@ cd $1; ./bootstrap
)
endef

define _ext_autogen_sh
$(eval
$1/configure: | $1
	@ cd $1; ./autogen.sh
)
endef

define _ext_configure
$(eval
$1/config.log: $1/configure
	@ cd $1; ./configure $($(subst -,_,$(subst ' ',_,$(subst .,_,$1)))_CONFIGURE_ARGS)
$1/Makefile: $1/config.log
	@ $(MAKE) $(AM_MAKEFLAGS) -C $1 Makefile
)
endef


define _ext_cmake
$(eval
.PHONY: $1/CMakeCache.txt
$1/CMakeLists.txt: | $1
$1/CMakeCache.txt: $1/CMakeLists.txt
	@ cd $1; cmake $($(subst -,_,$(subst ' ',_,$(subst .,_,$1)))_CMAKE_ARGS) .
$1/Makefile: $1/CMakeCache.txt
)
endef


define _ext_cmake_offsrc
$(eval
.PHONY: $3/CMakeCache.txt
$2/CMakeLists.txt: | $2
$3/CMakeCache.txt: $2/CMakeLists.txt | $3
	@ cd $3; cmake $($(subst -,_,$(subst ' ',_,$(subst .,_,$1)))_CMAKE_ARGS) $(abspath $2)
$2/Makefile: $3/CMakeCache.txt
)
endef


##           ##
##   CMAKE   ##
##           ##

define _ext_cmake
$(eval
.PHONY: $1/CMakeCache.txt
$1/CMakeLists.txt: | $1
$1/CMakeCache.txt: $1/CMakeLists.txt
	@ cd $1; cmake $($(subst -,_,$(subst ' ',_,$(subst .,_,$1)))_CMAKE_ARGS) .
$1/Makefile: $1/CMakeCache.txt
)
endef


define _ext_cmake_offsrc
$(eval
.PHONY: $3/CMakeCache.txt
$2/CMakeLists.txt: | $2
$3/CMakeCache.txt: $2/CMakeLists.txt | $3
	@ cd $3; cmake $($(subst -,_,$(subst ' ',_,$(subst .,_,$1)))_CMAKE_ARGS) $(abspath $2)
$2/Makefile: $3/CMakeCache.txt
)
endef




ext-: ##@ext enter ext module with $TARGET
ext-%: %/Makefile
	@ $(MAKE) $(AM_MAKEFLAGS) -C $(subst ext-,,$@) $(TARGET)

all: TARGET = all
all: $(EXT_MODULES)

# clean: TARGET = clean
# clean: ext-libjit


list: ##@ext list defined ext modules
list : _item = $(info | $1: ) $(foreach x,$($1),$(info |  - $x))
list:
	@ \
	$(info ,-----------------------------------------------------------------) \
	$(info | ext modules defined ) \
	$(info |) \
	$(call _item,EXT_MODULES) \
	$(info |) \
	$(info |) \
	$(info `-----------------------------------------------------------------) :



# /////////////////////////////////////////////////////////////////////////// //
# //  EXT MODULES  ////////////////////////////////////////////////////////// //
# /////////////////////////////////////////////////////////////////////////// //

DOWNLOADS  = $(EXT_MODULES)

# ccStickers: ##@ext download ccStickers sources
# ccStickers_URL = $(CCSTICKERS_URL)
# $(call _ext_bootstrap,ccStickers)
# $(call _ext_configure,ccStickers)

# libjit: ##@ext download libjit sources
# libjit_URL = $(LIBJIT_URL)
# $(call _ext_bootstrap,libjit)
# $(call _ext_configure,libjit)

# llvm:   ##@ext download llvm sources
# llvm_URL   = $(LLVM_URL)

ext-nodejs-node: AM_MAKEFLAGS += -j$(MAKE_PROCESS)
nodejs-node:
nodejs_node_URL = $(NODEJS_NODE_URL)
nodejs_node_CONFIGURE_ARGS = --shared
$(call _ext_configure,nodejs-node)


EXT_MDSPLUS_URL    ?= https://github.com/mdsplus/mdsplus.git
EXT_MDSPLUS_BRANCH ?= alpha
ext-mdsplus: AM_MAKEFLAGS += -j1
mdsplus: ##@ext mdsplus
mdsplus_URL    = $(EXT_MDSPLUS_URL)
mdsplus_BRANCH = $(EXT_MDSPLUS_BRANCH)
$(call _ext_bootstrap,mdsplus)
$(call _ext_configure,mdsplus)





# /////////////////////////////////////////////////////////////////////////// //
# //  DOCKER RPM BUILDER  /////////////////////////////////////////////////// //
# /////////////////////////////////////////////////////////////////////////// //

DOWNLOADS += docker-rpm-builder
docker-rpm-builder: ##@@rpm docker-rpm-builder download code
docker_rpm_builder_URL = https://github.com/docker-rpm-builder/docker-rpm-builder.git
docker_rpm_builder_BRANCH = 1.42

docker-rpm-builder-install: ##@rpm install local docker-rpm-builder package
docker-rpm-builder-install: | docker-rpm-builder
	@ cd docker-rpm-builder; python2 setup.py install --user







##    .########.########.##....##..######...#######..########..########.##........#######..##......##  
##    ....##....##.......###...##.##....##.##.....##.##.....##.##.......##.......##.....##.##..##..##  
##    ....##....##.......####..##.##.......##.....##.##.....##.##.......##.......##.....##.##..##..##  
##    ....##....######...##.##.##..######..##.....##.########..######...##.......##.....##.##..##..##  
##    ....##....##.......##..####.......##.##.....##.##...##...##.......##.......##.....##.##..##..##  
##    ....##....##.......##...###.##....##.##.....##.##....##..##.......##.......##.....##.##..##..##  
##    ....##....########.##....##..######...#######..##.....##.##.......########..#######...###..###.  

DOWNLOADS += tensorflow

tensorflow: ##@ext tensorflow
tensorflow_URL    = https://github.com/tensorflow/tensorflow.git
tensorflow_BRANCH = master


DOCKER_TARGETS = tf-%
tf-init: ##@tf build process
tf-init: DOCKER_IMAGE = tensorflow/tensorflow:devel-py3
tf-init: DOCKER_CONTAINER = tf-build
tf-init: DOCKER_MOUNTS = dist-packages:/usr/local/lib/python3.6/dist-packages
tf-init: PYTHON_PACKAGES = pip six numpy wheel mock # python packages needed for tf
tf-init: | tensorflow pip-install
	@ cd tensorflow;

tf-%: DOCKER_CONTAINER = tf-build

tensorflow/.tf_configure.bazelrc: | tf-init
	@ cd tensorflow; ./configure

tf-configure: tensorflow/.tf_configure.bazelrc

tf-build: ##@tf build tensorflow from sources in docker container
tf-build: tf-configure | tf-init tensorflow
	@ cd tensorflow; bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

tf-shell: tf-init
	@ bash



# define _install_vscode =
#  apt-get update; \
#  apt install -y gpg-agent software-properties-common apt-transport-https git wget libasound2 pylint; \
#  apt install -y libasound2 git; \
#  apt install -y python3-tk pylint; \
#  wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | apt-key add -; \
#  add-apt-repository -ru "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"; \
#  add-apt-repository -u "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"; \
#  apt install -y code;
# endef

# tf-install-vscode: USER=root
# tf-install-vscode: ##@@tf install vscode inside docker container (not working in docker-targets)
# tf-install-vscode:
# 	@ $(_install_vscode)

# tf-install-mdsplus: USER=root
# tf-install-mdsplus: ##@tf install mdsplus in container
# tf-install-mdsplus:
# 	@ wget http://www.mdsplus.org/dist/mdsplus.gpg.key; \
# 	  apt-key add mdsplus.gpg.key; \
# 	  add-apt-repository -ru "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus Alpha"; \
# 	  add-apt-repository -u "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus Alpha"; \
# 	  apt install -y mdsplus-alpha; \
# 	  cd /usr/local/mdsplus/mdsobjects/python/; \
# 	  python3 setup.py install;









##        ..######..########..########......#######..########.....########.##.....##....###....##.....##.########..##.......########
##        .##....##.##.....##.##.....##....##.....##.##.....##....##........##...##....##.##...###...###.##.....##.##.......##......
##        .##.......##.....##.##.....##....##.....##.##.....##....##.........##.##....##...##..####.####.##.....##.##.......##......
##        .##.......########..########.....##.....##.########.....######......###....##.....##.##.###.##.########..##.......######..
##        .##.......##........##...........##.....##.##...........##.........##.##...#########.##.....##.##........##.......##......
##        .##....##.##........##...........##.....##.##...........##........##...##..##.....##.##.....##.##........##.......##......
##        ..######..##........##............#######..##...........########.##.....##.##.....##.##.....##.##........########.########


DOWNLOADS += cppop-example
cppop-example: ##@cppop download example
cppop_example_URL = https://github.com/andrearigoni/tensorflow-cpp-op-example.git
cppop_example_BRANCH = tf-2.0
cppop_example_DIR = cppop-example

DOCKER_TARGETS += cppop-example/Makefile
cppop-example/Makefile: DOCKER_CONTAINER = tf-build
$(call _ext_cmake,cppop-example)

DOCKER_TARGETS += ext-cppop-example
ext-cppop-example: ##@cppop compile example
ext-cppop-example: DOCKER_CONTAINER = tf-build
ext-cppop-example: tf-init cppop-example/Makefile | cppop-example
	@ make -C cppop-example all

DOCKER_TARGETS += ext-cppop-example-test
ext-cppop-example-test: ##@cppop run tests
ext-cppop-example-test: DOCKER_CONTAINER = tf-build
ext-cppop-example-test: ext-cppop-example
	@ cd cppop-example; \
	  LD_LIBRARY_PATH=$$(pwd) python3 inner_product_tests.py







