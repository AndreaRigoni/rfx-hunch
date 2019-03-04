include $(top_srcdir)/Common.mk

PYTHON_PACKAGES = numpy




##    .########.########.##....##..######...#######..########..########.##........#######..##......##  
##    ....##....##.......###...##.##....##.##.....##.##.....##.##.......##.......##.....##.##..##..##  
##    ....##....##.......####..##.##.......##.....##.##.....##.##.......##.......##.....##.##..##..##  
##    ....##....######...##.##.##..######..##.....##.########..######...##.......##.....##.##..##..##  
##    ....##....##.......##..####.......##.##.....##.##...##...##.......##.......##.....##.##..##..##  
##    ....##....##.......##...###.##....##.##.....##.##....##..##.......##.......##.....##.##..##..##  
##    ....##....########.##....##..######...#######..##.....##.##.......########..#######...###..###.  

DOWNLOADS = tensorflow

tensorflow: ##@ext tensorflow
tensorflow_URL    = https://github.com/tensorflow/tensorflow.git
tensorflow_BRANCH = master

# python packages needed for tf
PYTHON_PACKAGES += pip six numpy wheel mock

DOCKER_TARGETS = tf-%
tf-init: ##@tf build process
tf-init: DOCKER_IMAGE = tensorflow/tensorflow:nightly-devel-py3
tf-init: DOCKER_CONTAINER = tf-build
tf-init: DOCKER_MOUNTS = dist-packages:/usr/local/lib/python3.6/dist-packages/tensorflow
tf-init: | tensorflow pip-install
	@ cd tensorflow;

tf-%: DOCKER_CONTAINER = tf-build

tensorflow/.tf_configure.bazelrc: | tf-init
	@ cd tensorflow; ./configure

tf-configure: tensorflow/.tf_configure.bazelrc

tf-build: tf-configure | tf-init
	@ cd tensorflow; bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

tf-shell: tf-init
	@ bash




define _install_vscode =
 apt-get update; \
 apt install -y gpg-agent software-properties-common apt-transport-https git wget libasound2 pylint; \
 apt install -y libasound2 git; \
 apt install -y python3-tk pylint; \
 wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | apt-key add -; \
 add-apt-repository -ru "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"; \
 add-apt-repository -u "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"; \
 apt install -y code;
endef

tf-dev-init: USER=root
tf-dev-init: ##@tf install other deps inside docker container
tf-dev-init:
	@ $(_install_vscode)

tf-mdsplus-init: USER=root
tf-mdsplus-init: ##@tf install mdsplus in container
tf-mdsplus-init:
	@ wget http://www.mdsplus.org/dist/mdsplus.gpg.key; \
	  apt-key add mdsplus.gpg.key; \
		add-apt-repository -ru "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus Alpha"; \
	  add-apt-repository -u "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus Alpha"; \
	  apt install -y mdsplus-alpha; \
		cd /usr/local/mdsplus/mdsobjects/python/; \
		python3 setup.py install;


DOCKER_TARGETS += tf-edit
tf-edit:
	@ $(MAKE) edit-code







##
## APP
##

DOCKER_TARGETS += read

PYSCRIPTS = read_db.py
read: ##@tprofile read data
read: DOCKER_CONTAINER = tf-build
read: $(PYSCRIPTS) pip-install
	@ python $<;







