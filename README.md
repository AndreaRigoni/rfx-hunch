# RFX-Hunch

further documentation is available in [USER MANUAL](doc/README.md)



This is a autoconf/automake minimal setup to start a new project. It uses Kconfig integration 
provided by https://github.com/AndreaRigoni/autoconf-kconfig as a submodule in conf/kconfig

In the following a easy setup procedure:

<pre>
git clone https://github.com/andrearigoni/autoconf-bootstrap.git
cd autoconf-bootstrap
./bootstrap
mkdir build
cd build
../configure --enable-kconfig
# enjoy
</pre>

