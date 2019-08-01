# note: it must be an ubuntu like image (see default)
# ARG  DOCKER_IMAGE=tensorflow/tensorflow:nightly-devel-py3
FROM tensorflow/tensorflow:2.0.0b0-py3

RUN apt-get update;

# cmake is needed by cpp-op-example (build custom operators)
RUN apt-get install -y cmake

#avoid tzdata interactive shell to ask for region code
ENV DEBIAN_FRONTEND=noninteractive

# MDSplus prereq
RUN apt-get update; \
    apt-get -y upgrade; \
    apt-get install -y rsync reprepro wget tar \
            libcurl4-gnutls-dev libreadline-dev \
            python-dev python-setuptools python-numpy

# fix (very bad) missing library
RUN ln -f -s /lib/x86_64-linux-gnu/libreadline.so.7 /lib/x86_64-linux-gnu/libreadline.so.6

# MDSplus install
RUN apt install -y gpg-agent software-properties-common apt-transport-https git wget libasound2 pylint; \
	wget http://www.mdsplus.org/dist/mdsplus.gpg.key; \
	apt-key add mdsplus.gpg.key; \
	add-apt-repository -ru "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus stable"; \
	add-apt-repository -u "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus stable"; \
	mkdir -p /usr/lib/python3.6/site-packages/; \
	apt install -y mdsplus mdsplus-python; \
	ldconfig;

# Import all MDSplus env for RFX experiment
COPY envsyms /usr/local/mdsplus/local/envsyms
COPY experimentsVersion /usr/local/mdsplus/local/experimentsVersion
COPY experimentsVersionPath /usr/local/mdsplus/local/experimentsVersionPath

# PYTHON SETUP MODULES
RUN apt install -y python3-tk

# install remote debugger (Python Tools for Visual Studio debug server)
RUN pip3 install ptvsd

# plot utils
RUN pip3 install seaborn

# statmodels
RUN pip3 install -U statsmodels


RUN apt install -y jupyter ipython

# make Jupyter able save to pdf
# RUN apt install -y pandoc pandoc-citeproc pandoc-data texlive-xetex

COPY treeSetSource /usr/bin/treeSetSource
RUN  chmod 555 /usr/bin/treeSetSource