include $(top_srcdir)/Common.mk


export PYTHON_PACKAGES
export PYTHONPATH
export PYTHONUSERBASE

PYTHON_PACKAGES = pip six numpy wheel mock # python packages needed for tf
PYTHONPATH = $(abs_top_builddir)/conf/python/site-packages/lib/python3.6/site-packages/:/usr/local/lib/python3.6/dist-packages:/usr/lib/python3.6/site-packages/:$(abs_srcdir)


##    .########.########.##....##..######...#######..########..########.##........#######..##......##  
##    ....##....##.......###...##.##....##.##.....##.##.....##.##.......##.......##.....##.##..##..##  
##    ....##....##.......####..##.##.......##.....##.##.....##.##.......##.......##.....##.##..##..##  
##    ....##....######...##.##.##..######..##.....##.########..######...##.......##.....##.##..##..##  
##    ....##....##.......##..####.......##.##.....##.##...##...##.......##.......##.....##.##..##..##  
##    ....##....##.......##...###.##....##.##.....##.##....##..##.......##.......##.....##.##..##..##  
##    ....##....########.##....##..######...#######..##.....##.##.......########..#######...###..###.  

DOWNLOADS = tensorflow

tensorflow: ##@tf download git tensorflow code (master)
tensorflow_URL    = https://github.com/tensorflow/tensorflow.git
tensorflow_BRANCH = master

DOCKER_TARGETS = tf-%
tf-%: DOCKER_CONTAINER = tf-build

#IMAGE: tensorflow/tensorflow:nightly-devel-py3
tf-init: ##@tf build process
tf-init: DOCKER_IMAGE = tf-devel-py3
tf-init: DOCKER_URL = $(srcdir)/docker
tf-init: DOCKER_MOUNTS = dist-packages:/usr/local/lib/python3.6/dist-packages
tf-init: DOCKER_PORTS = 8888
tf-init: pip-install | tensorflow 
	@ cd tensorflow; 

DOCKER_TARGETS += pip-install
pip-install: DOCKER_CONTAINER = tf-build

tensorflow/.tf_configure.bazelrc: | tf-init
	@ cd tensorflow; ./configure

tf-configure: ##@tf reconfigure src code project
tf-configure: tensorflow/.tf_configure.bazelrc

tf-build: ##@tf build tensorflow code (not needed)
tf-build: tf-configure | tf-init
	@ cd tensorflow; bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

tf-shell: ##@tf enter container with shell (bash)
tf-shell: tf-init
	@ bash -l

## ...##.##.....##.##.........###....########..########.
## ...##.##.....##.##........##.##...##.....##.##.....##
## .#########.#########.....##...##..##.....##.##.....##
## ...##.##.....##.##......##.....##.########..########.
## .#########.#########....#########.##........##.......
## ...##.##.....##.##......##.....##.##........##.......
## ...##.##.....##.##......##.....##.##........##.......

export abs_srcdir
export abs_builddir
export rfx_ver = inProgress


# automatically reload the following files (this works only if 'ipysh' has been imported too)
export IPY_AIMPORT = Hunch_utils models.AEFIT models.SPFIT

DOCKER_TARGETS += runpy
PYSCRIPTS = jpnb/STEP1_fit.py
runpy: ##@jp Run ipython console starting by script defined in $NAME or $PYSCRIPTS
runpy: DOCKER_CONTAINER = tf-build
runpy: $(NAME) $(PYSCRIPTS) | pip-install
	@ ipython -i $< $(ARGS)


DOCKER_TARGETS += jupyter
jupyter: DOCKER_CONTAINER = tf-build
jupyter: ##@jp start jupyter server on jpnb $(srcdir) folder
	@ jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=$(abs_srcdir)/jpnb



DOWNLOADS += rdipietro
rdipietro: ##download rdpietro notebooks
rdipietro_URL = https://github.com/rdipietro/jupyter-notebooks.git
rdipietro_BRANCH = master
rdipietro_DIR = $(srcdir)/jpnb/rdipietro









##   .########..########.########..########..########..######.....###....########.########.########.
##   .##.....##.##.......##.....##.##.....##.##.......##....##...##.##......##....##.......##.....##
##   .##.....##.##.......##.....##.##.....##.##.......##........##...##.....##....##.......##.....##
##   .##.....##.######...########..########..######...##.......##.....##....##....######...##.....##
##   .##.....##.##.......##........##...##...##.......##.......#########....##....##.......##.....##
##   .##.....##.##.......##........##....##..##.......##....##.##.....##....##....##.......##.....##
##   .########..########.##........##.....##.########..######..##.....##....##....########.########.

tf-install-vscode: USER=root
tf-install-vscode: ##@@tf install vscode inside docker container (not working in docker-targets)
tf-install-vscode:
	@ \
	 apt-get update; \
	 apt install -y gpg-agent software-properties-common apt-transport-https git wget libasound2 pylint; \
	 apt install -y libasound2 git; \
	 apt install -y python3-tk pylint; \
	 wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | apt-key add -; \
	 add-apt-repository -ru "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"; \
	 add-apt-repository -u "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"; \
	 apt install -y code;


tf-install-mdsplus: export PYTHONUSERBASE=/usr/local/lib/python3.6/dist-packages/
tf-install-mdsplus: USER=root
tf-install-mdsplus: ##@tf install mdsplus in container
tf-install-mdsplus:
	@ \
	  apt install -y gpg-agent software-properties-common apt-transport-https git wget libasound2 pylint; \
	  wget http://www.mdsplus.org/dist/mdsplus.gpg.key; \
	  apt-key add mdsplus.gpg.key; \
	  add-apt-repository -ru "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus stable"; \
	  add-apt-repository -u "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus stable"; \
	  mkdir -p /usr/lib/python3.6/site-packages/; \
	  apt install -y mdsplus mdsplus-python;


tf-install-mdsplus-alpha: USER=root
tf-install-mdsplus-alpha: ##@tf install mdsplus-alpha in container
tf-install-mdsplus-alpha:
	@ \
	  apt install -y gpg-agent software-properties-common apt-transport-https git wget libasound2 pylint; \
	  wget http://www.mdsplus.org/dist/mdsplus.gpg.key; \
	  apt-key add mdsplus.gpg.key; \
	  add-apt-repository -ru "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus alpha"; \
	  add-apt-repository -u "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus alpha"; \
	  mkdir -p /usr/lib/python3.6/site-packages/; \
	  apt install -y mdsplus-alpha mdsplus-alpha-python;


