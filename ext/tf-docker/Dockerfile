# note: it must be an ubuntu like image (see default)
ARG  DOCKER_IMAGE=tensorflow/tensorflow:nightly-devel-py3
FROM ${DOCKER_IMAGE}

RUN apt-get update;

# cmake is needed by cpp-op-example (build custom operators)
RUN apt-get install -y cmake




