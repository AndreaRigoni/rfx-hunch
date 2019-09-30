from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import Hunch_utils as Htils

from Dataset_QSH import *

import tensorflow as tf
import abc




class SNEPlot():
    __metaclass__ = abc.ABCMeta    

    class CfgDict(dict):
        ''' A class to overload a dictionary with direct access to keys as internal methods
        '''
        def __init__(self, *args, **kwargs):
            super(SNEPlot.CfgDict, self).__init__(*args, **kwargs)
            self.__dict__ = self        
        def __add__(self, other):
            super(SNEPlot.CfgDict, self).update(other)
            return self


    def __init__(self, *argv, **argd):        
        self._cfg = SNEPlot.CfgDict({
            'sample_size': 100,
        }) + argd
        
    def set_model(self, model):
        self._model = model

    def set_data(self, data):        
        assert isinstance(data, Dataset_QSH), "Input variables should be QSH dataset"
        self._data = data    
    

class tSNE_PlotBokeh(SNEPlot):
    from bokeh.io import show, output_notebook    
    from bokeh import events
    from bokeh.models import CustomJS, Div, Button, Slider
    from bokeh.models import CustomJS, ColumnDataSource, Slider, TextInput, HoverTool
    from bokeh.layouts import column, row
    from bokeh.plotting import figure

    ## Events with attributes
    point_attributes = ['x', 'y', 'sx', 'sy']                  # Point events
    wheel_attributes = point_attributes + ['delta']            # Mouse wheel event
    pan_attributes = point_attributes + ['delta_x', 'delta_y'] # Pan event
    pinch_attributes = point_attributes + ['scale']            # Pinch event

    # events.Tap, events.DoubleTap, events.Press,
    # events.PanStart, events.PanEnd, events.PinchStart, events.PinchEnd,
    
    point_events = [
        events.MouseMove, events.MouseEnter, events.MouseLeave,
    ]


    def __init__(self, *argv, **argd):
        super(tSNE_PlotBokeh,self).__init__(*argv,**argd)        
        self._figure_ls = tSNE_PlotBokeh.figure(plot_width=400, plot_height=400,tools="lasso_select,pan,zoom_in,zoom_out,reset")
        self._figure_gn = tSNE_PlotBokeh.figure(plot_width=400, plot_height=400,tools="pan,zoom_in,zoom_out,reset")
        self._figure_pgn = tSNE_PlotBokeh.figure(plot_width=400, plot_height=400,tools="pan,zoom_in,zoom_out,reset")
        self._div = tSNE_PlotBokeh.Div(width=200, height=400, height_policy="fixed")
        self._layout = tSNE_PlotBokeh.row(self._figure_ls, self._figure_pgn)  # , self._div

        # LS PLOT
        self._data_ls = tSNE_PlotBokeh.ColumnDataSource(data=dict(x=[],y=[],id=[]))
        scatter_ls = self._figure_ls.circle('x','y', size=10, source=self._data_ls, alpha=0.4, hover_color='olive', hover_alpha=1.0)

        # GN PLOT        
        self._data_gn  = tSNE_PlotBokeh.ColumnDataSource(data=dict(x=[],y=[]))
        self._data_gn2 = tSNE_PlotBokeh.ColumnDataSource(data=dict(x=[],y=[]))
        self._figure_gn.multi_line('x','y', source=self._data_gn)
        self._figure_gn.multi_line('x','y', source=self._data_gn2, color='red')

        self._data_pgn  = tSNE_PlotBokeh.ColumnDataSource(data=dict(x=[],y=[]))
        self._data_pgn2 = tSNE_PlotBokeh.ColumnDataSource(data=dict(x=[],y=[]))
        self._figure_pgn.multi_line('x','y', source=self._data_pgn)
        self._figure_pgn.multi_line('x','y', source=self._data_pgn2, color='red')

        # trace     
        self._inx = tSNE_PlotBokeh.TextInput(value='')
        def inx_cb(attr, old, new):
            ids = [int(x.strip()) for x in new.split()]
            self._plot_cb(ids, self._data_gn,  axis='rho')
            self._plot_cb(ids, self._data_pgn, axis='prel')
        self._inx.on_change('value',inx_cb)

        self._inh = tSNE_PlotBokeh.TextInput(value='')
        def inh_cb(attr, old, new):
            ids = [int(x.strip()) for x in new.split()]
            self._plot_cb(ids, self._data_gn2, axis='rho')
            self._plot_cb(ids, self._data_pgn2, axis='prel')
        self._inh.on_change('value',inh_cb)
        
        # EVENTS CONNECTIONS
        self._data_ls.selected.js_on_change('indices', self.selected_event())
        self._figure_ls.add_tools(tSNE_PlotBokeh.HoverTool(tooltips=None, 
          callback=self.display_event(self._div, attributes=tSNE_PlotBokeh.point_attributes),
          renderers=[scatter_ls]))
    
    def plot(self, notebook_url='http://172.17.0.2:8888'):
        def plot(doc):
            doc.add_root(self._layout)
        tSNE_PlotBokeh.show(plot, notebook_url=notebook_url)

    def set_data(self, data, counts=200):
        super(tSNE_PlotBokeh, self).set_data(data)
        self._counts=counts
        md = self._model
        ds = self._data._dataset[0:counts]
        xy = md(( np.concatenate([ds['prel'],ds['te']], axis=1) ))
        ds_dict = dict(x=xy[:,0],y=xy[:,1],id=range(counts),prel=ds['prel'],te=ds['te'])
        self._data_ls.data = ds_dict

    def _plot_cb(self, ids, data, axis='prel'):
        qsh = self._data
        ds = self._data._dataset[0:self._counts]
        x = [ qsh.clean_array(ds[i][axis]) for i in ids ]
        y = [ qsh.clean_array(ds[i]['te']) for i in ids ]
        if len(x) > 0:
            data.data = dict(x=x,y=y)
        
    def selected_event(self, attributes=[]):
        return tSNE_PlotBokeh.CustomJS(args=dict(inx=self._inx, data_ls=self._data_ls), code="""
            var inds = data_ls.selected.indices;
            var hdata = data_ls.data
            var inx_str = ""
            for (var i = 0; i < inds.length; i++) {
                var id = inds[i]
                var x = hdata.x[id]
                var y = hdata.y[id]            
                var d = hdata.id[id]
                inx_str += Number(d) + " "                                
            }
            inx.value = inx_str
        """)

    def display_event(self, div, attributes=[], style = 'float:left;clear:left;font_size=10pt'):
        return tSNE_PlotBokeh.CustomJS(args=dict(div=div, inh=self._inh, data_ls=self._data_ls, data_gn=self._data_gn), code="""
            var attrs = %s; 
            var hdata = data_ls.data;
            var gdata = data_gn.data;
            var line = ""            
            var inh_str = ""
            var indices = cb_data.index['1d'].indices;                        
            for (var i = 0; i < indices.length; i++) {
                var id = indices[i]
                var x = hdata.x[id]
                var y = hdata.y[id]            
                var d = hdata.id[id]
                line += "<span style=%r>" + Number(d) + ": (" 
                                                + Number(x).toFixed(3) + ","
                                                + Number(y).toFixed(3) + 
                                ")</span>\\n";                
                inh_str += Number(d) + " "                
                // gdata.x = hdata.prel[id];
                // gdata.y = hdata.te[id];
                
            }
            inh.value = inh_str
            var text = div.text.concat(line);
            var lines = text.split("\\n")
            if (lines.length > 15)
                lines.shift();
            div.text = lines.join("\\n");
        """ % (attributes, style))



