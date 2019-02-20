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

clean: TARGET = clean
clean: ext-libjit


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







##    .########.########....########..##.....##.####.##.......########.
##    ....##....##..........##.....##.##.....##..##..##.......##.....##
##    ....##....##..........##.....##.##.....##..##..##.......##.....##
##    ....##....######......########..##.....##..##..##.......##.....##
##    ....##....##..........##.....##.##.....##..##..##.......##.....##
##    ....##....##..........##.....##.##.....##..##..##.......##.....##
##    ....##....##..........########...#######..####.########.########.


# BAZEL_CACHEDIR = /tmp/bazel_cache
# DIRECTORIES  = $(BAZEL_CACHEDIR)
# LOCAL_DIRS   = $(BAZEL_CACHEDIR)
# export BAZEL_CACHEDIR

DOWNLOADS += tensorflow
tensorflow: ##@ext tensorflow
tensorflow_URL    = https://github.com/tensorflow/tensorflow.git
tensorflow_BRANCH = master

# python packages needed for tf
PYTHON_PACKAGES = pip six numpy wheel mock
tf_DOCKER_IMAGE = tensorflow/tensorflow:nightly-devel-py3

DOCKER_TARGETS = tf-%

tf-%: DOCKER_CONTAINER = tf-build

tf-init: ##@tf init docker container for build process
tf-init: DOCKER_IMAGE = $(tf_DOCKER_IMAGE)
tf-init: DOCKER_URL   = $(srcdir)/tf-docker
tf-init: | tensorflow pip-install
	@ cd tensorflow;

tensorflow/.tf_configure.bazelrc: tf-init
	@ cd tensorflow; ./configure

tf-configure: tensorflow/.tf_configure.bazelrc

tf-build: ##@tf build tensorflow from sources
tf-build: tf-configure tf-init
	@ cd tensorflow; \
	  bazel build --remote_http_cache=http://localhost:8980 \
	   --spawn_strategy=standalone \
	   --config=opt //tensorflow/tools/pip_package:build_pip_package

tf-shell: ##@tf enter container shell
tf-shell: | tf-init
	@ /bin/bash




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
cppop_example_BRANCH = master
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
ext-cppop-example: ##@cppop run tests
ext-cppop-example-test: DOCKER_CONTAINER = tf-build
ext-cppop-example-test: ext-cppop-example
	@ cd cppop-example; \
	  LD_LIBRARY_PATH=$$(pwd) python3 inner_product_tests.py

