from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Hunch_utils as Htils
import models.AEFIT as AEFIT
import Dummy_g1data

import numpy as np
import tensorflow as tf
import abc



class LSPlot():
    __metaclass__ = abc.ABCMeta    

    class CfgDict(dict):
        ''' A class to overload a dictionary with direct access to keys as internal methods
        '''
        def __init__(self, *args, **kwargs):
            super(LSPlot.CfgDict, self).__init__(*args, **kwargs)
            self.__dict__ = self        
        def __add__(self, other):
            super(LSPlot.CfgDict, self).update(other)
            return self


    def __init__(self, *argv, **argd):        
        self._cfg = LSPlot.CfgDict({
            'sample_size': 100,
        }) + argd

    def set_model(self, model):
        # assert isinstance(model, AEFIT.Hunch_VAE), "Input variables should be AEFIT"
        self._model = model
        
    def set_data(self, data):
        # assert isinstance(data, Dummy_g1data.FiniteSequence1D), "Input variables should be FiniteSequence1D"
        self._data = data    
    


class LSPlotBokeh(LSPlot):
    import numpy as np
    from bokeh.io import show, output_file, output_notebook
    from bokeh import events
    from bokeh.models import CustomJS, Div, Button, Slider
    from bokeh.models import CustomJS, ColumnDataSource, Slider, TextInput    
    from bokeh.layouts import column, row
    from bokeh.plotting import figure

    ## Events with attributes
    point_attributes = ['x', 'y', 'sx', 'sy']                  # Point events
    wheel_attributes = point_attributes + ['delta']            # Mouse wheel event
    pan_attributes = point_attributes + ['delta_x', 'delta_y'] # Pan event
    pinch_attributes = point_attributes + ['scale']            # Pinch event

    point_events = [
        events.Tap, events.DoubleTap, events.Press,
        events.MouseMove, events.MouseEnter, events.MouseLeave,
        events.PanStart, events.PanEnd, events.PinchStart, events.PinchEnd,
    ]


    def __init__(self, *argv, **argd):
        super(LSPlotBokeh,self).__init__(*argv,**argd)
        self._figure_ls = LSPlotBokeh.figure(plot_width=400, plot_height=400,tools="pan,zoom_in,zoom_out,reset,crosshair")
        self._figure_gn = LSPlotBokeh.figure(plot_width=400, plot_height=400,tools="pan,zoom_in,zoom_out,reset",y_range=(0,1))
        self._div = LSPlotBokeh.Div(width=200, height=400, height_policy="fixed")
        self._layout = LSPlotBokeh.row(self._figure_ls,self._figure_gn, self._div)        

        # trace mouse position        
        self._inx = LSPlotBokeh.TextInput(value='')
        self._pos = LSPlotBokeh.ColumnDataSource(data=dict(x=[0],y=[0],dim=[0]))        
        def posx_cb(attr, old, new):
            pos = self._pos
            x,y = [float(x.strip()) for x in new.split(',')]
            pos.data['x'][0] = x
            pos.data['y'][0] = y
            self.plot_generative(x,y)
        self._inx.on_change('value',posx_cb)
        
        # LS PLOT
        self._data_ls = LSPlotBokeh.ColumnDataSource(data=dict(x=[],y=[]))
        self._figure_ls.scatter('x','y',source=self._data_ls)
        for event in LSPlotBokeh.point_events:
            self._figure_ls.js_on_event(event, self.display_event(self._div, attributes=LSPlotBokeh.point_attributes))

        # NG PLOT        
        self._data_gn = LSPlotBokeh.ColumnDataSource(data=dict(x=[],y=[]))
        self._figure_gn.line('x','y', source=self._data_gn)
        self._figure_gn.scatter('x','y', source=self._data_gn)
    
    def plot(self, notebook_url='http://172.17.0.2:8888'):
        def plot(doc):
            doc.add_root(self._layout)
        LSPlotBokeh.show(plot, notebook_url=notebook_url)

    def set_data(self, data, counts=200):
        super(LSPlotBokeh, self).set_data(data)
        md = self._model
        ds = self._data
        self._cold = []
        if isinstance(ds, Dummy_g1data.Dummy_g1data):
            # from bokeh.palettes import Category10
            # import itertools            
            # colors = itertools.cycle(Category10[10])
            dx = 1/ds._size
            x = np.linspace(0+dx/2,1-dx/2,ds._size)  # spaced x axis
            for i,k in enumerate(ds.kinds,0):
                xy,_ = ds.gen_pt(id=0, x=x, kind=i)
                self._cold.append( LSPlotBokeh.ColumnDataSource(data=dict(x=xy[:,0],y=xy[:,1]))  )
                self._figure_gn.line('x','y',source=self._cold[i], line_width=3, line_alpha=0.6, color='red')

        ds = self._data.ds_array.prefetch(counts).batch(counts)        
        ts,_ = ds.make_one_shot_iterator().get_next()
        if md.latent_dim == 2:
            m,v = md.encode(ts)
            # z = md.reparameterize(m,v)
            data=dict( x=m[:,0].numpy().tolist(), y=m[:,1].numpy().tolist() )
            self._data_ls.data = data

    def plot_generative(self, x, y):
        md = self._model
        XY = md.decode(tf.convert_to_tensor([[x,y]]), apply_sigmoid=True)
        X,Y = tf.split(XY[0], 2)
        data = dict( x=X.numpy().tolist(), y=Y.numpy().tolist() )
        self._data_gn.data = data

    def display_event(self, div, attributes=[], style = 'float:left;clear:left;font_size=10pt'):
        "Build a suitable CustomJS to display the current event in the div model."
        return LSPlotBokeh.CustomJS(args=dict(div=div, pos=self._pos, inx=self._inx), code="""
            var attrs = %s; var args = [];
            for (var i = 0; i<attrs.length; i++)
                args.push(attrs[i] + '=' + Number(cb_obj[attrs[i]]).toFixed(2));
            var line = ""
            var x = cb_obj[attrs[0]]
            var y = cb_obj[attrs[1]]
            // if (cb_obj.event_name == "tap")
            //    code = "tap"
            inx.value = Number(x).toFixed(3) + "," + Number(y).toFixed(3)                
            line += "<span style=%r><b>" + cb_obj.event_name + "</b>(" + Number(x).toFixed(3) + ","
                                                                          + Number(y).toFixed(3) + 
                                                                          ")</span>\\n";
            var text = div.text.concat(line);
            var lines = text.split("\\n")
            if (lines.length > 15)
                lines.shift();
            div.text = lines.join("\\n");
        """ % (attributes, style))






