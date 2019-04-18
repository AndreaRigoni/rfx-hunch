
class Bootstrap_support():

    @staticmethod
    def autoreload(modules=None, env='IPY_AIMPORT'):
        '''
        Set Autoreload of modules that are defined in IPY_AIMPORT env variable
        This works for IPython 3.x or higher
        '''
        try:
            import os, importlib
            import IPython.core
            ipy = IPython.core.getipython.get_ipython()
            ipy.magic("load_ext autoreload")
            ipy.magic("autoreload 1")
        except:
            print('could not load requred modules')
            pass
        if modules is None:
            modules = os.environ.get(env).split(" ")
        for m in modules:
            try:
                # if m is not '__main__':
                #     importlib.import_module(m)
                ipy.magic('aimport %s'%m)
                print('reload set for module ',m)
            except:
                pass
        return ipy


    @staticmethod
    def debug(wait=False):
        try:
            import ptvsd
            # Allow other computers to attach to ptvsd at this IP address and port.
            ptvsd.enable_attach(address=('*', 3000), redirect_output=True)
            # Pause the program until a remote debugger is attached
            if wait:
                ptvsd.wait_for_attach()
            print('debug active on *:3000')
        except:
            print('unable to set vs debugger')

import os
abs_srcdir   = os.environ.get('abs_srcdir')
abs_builddir = os.environ.get('abs_builddir')


# trigger autoreload functionality
Bootstrap_support.autoreload()

# trigger debug session
# Bootstrap_support.debug()

