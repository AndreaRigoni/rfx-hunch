from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Hunch_utils as Htils
import Dummy_g1data

import numpy as np
import tensorflow as tf
import abc

import models

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
        self._model = None
        self._data  = None


    def set_model(self, model):
        # assert isinstance(model, AEFIT.Hunch_VAE), "Input variables should be AEFIT"
        self._model = model
        
    def set_data(self, data):
        # assert isinstance(data, Dummy_g1data.FiniteSequence1D), "Input variables should be FiniteSequence1D"
        self._data = data    
    


class LSPlotBokeh(LSPlot):
    from bokeh.io import show, output_notebook, push_notebook
    from bokeh import events
    from bokeh.models import CustomJS, Div, Button, Slider, Toggle
    from bokeh.models import CustomJS, ColumnDataSource, Slider, TextInput, Range1d  
    from bokeh.layouts import column, row
    from bokeh.plotting import figure
    from bokeh.document import without_document_lock

    from bokeh.models import (
        LinearColorMapper,
        LogColorMapper,
    )
    from bokeh.palettes import PuBu, OrRd, RdBu, Category20

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
        self._target = None
        self._doc = None

        self._figure_ls = LSPlotBokeh.figure(plot_width=400, plot_height=400,tools="pan,box_zoom,zoom_in,zoom_out,reset,crosshair")
        self._figure_gn = LSPlotBokeh.figure(plot_width=400, plot_height=400,tools="pan,zoom_in,zoom_out,reset",x_range=(0,1),y_range=(0,1))
        self._div = LSPlotBokeh.Div(width=800, height=10, height_policy="fixed")        

        # trace mouse position
        self._inx = LSPlotBokeh.TextInput(value='')
        self._pos = LSPlotBokeh.ColumnDataSource(data=dict(x=[0],y=[0],dim=[0]))        
        def posx_cb(attr, old, new):
            pos = self._pos
            x,y = [float(x.strip()) for x in new.split(',')]
            pos.data['x'][0] = x
            pos.data['y'][0] = y
            self._doc.add_next_tick_callback(lambda: self.plot_generative(x,y))            
        self._inx.on_change('value',posx_cb)
        
        # COLOR MAPPERS
        self._mapper1 = LSPlotBokeh.LinearColorMapper(palette=LSPlotBokeh.PuBu[9], low=0, high=1)
        self._mapper2 = LSPlotBokeh.LinearColorMapper(palette=LSPlotBokeh.OrRd[9], low=0, high=1)
        self._mapper3 = LSPlotBokeh.LinearColorMapper(palette=LSPlotBokeh.RdBu[9], low=0, high=1)
        self._mapper3 = LSPlotBokeh.LinearColorMapper(palette=LSPlotBokeh.RdBu[9], low=0, high=1)
        self._mapper4 = LSPlotBokeh.LinearColorMapper(palette=LSPlotBokeh.Category20[20], low=0, high=20)


        # LS PLOT
        self._ls_glyphs = []
        def toggle_ls_glyphs(name = None):
            for g in self._ls_glyphs: g.visible = False                
            if name: self._figure_ls.select(name=name).visible = True

        self._data_ls = LSPlotBokeh.ColumnDataSource(data=dict(mx=[],my=[],vx=[],vy=[],zx=[],zy=[],tcentro=[],label=[]) )
        self._figure_ls.scatter('zx','zy',name='sample', legend='sample', size=10, source=self._data_ls, color='grey', alpha=0.2, line_width=0)        
        self._ls_glyphs += [self._figure_ls.circle('mx','my',name='Tc', legend='Tc',
                                                        size=10, 
                                                        source=self._data_ls, 
                                                        alpha=0.5, 
                                                        line_width=0,
                                                        fill_color={'field': 'tcentro', 'transform': self._mapper3}
                                                        )]
        self._ls_glyphs += [self._figure_ls.circle('mx','my',name='label', legend='label',
                                                        size=10, 
                                                        source=self._data_ls, 
                                                        alpha=0.5, 
                                                        line_width=0,
                                                        fill_color={'field': 'label', 'transform': self._mapper4}
                                                        )]
        self._figure_ls.legend[0].visible = False
        toggle_ls_glyphs(None)
        for event in LSPlotBokeh.point_events:
            self._figure_ls.js_on_event(event, self.display_event(self._div, attributes=LSPlotBokeh.point_attributes))


        # NG PLOT        
        self._data_gn = LSPlotBokeh.ColumnDataSource(data=dict(x=[],y=[]))
        self._figure_gn.line('x','y', source=self._data_gn)
        self._figure_gn.scatter('x','y', source=self._data_gn)

        # WIDGETS #
        self._b1 = LSPlotBokeh.Button(label="Update ls", button_type="success", width=150)
        self._b1.on_click(self.update_ls)
        self._b2 = LSPlotBokeh.Button(label="Plasma Tc", width=150)
        self._b2.on_click(lambda: toggle_ls_glyphs('Tc'))
        self._b3 = LSPlotBokeh.Button(label="label", width=150)
        self._b3.on_click(lambda: toggle_ls_glyphs('label'))

        self._layout = LSPlotBokeh.column( 
            LSPlotBokeh.row(self._figure_ls,self._figure_gn,
                LSPlotBokeh.column(
                    self._b1,
                    self._b2,
                    self._b3,
                )),
            #LSPlotBokeh.row(self._div)
        )
    
    def plot(self, notebook_url='http://172.17.0.2:8888'):
        self.plot_notebook(notebook_url)

    def plot_notebook(self, notebook_url='http://localhost:8888'):
        from bokeh.io import output_notebook
        output_notebook()
        def plot(doc):
            self._doc = doc
            doc.add_root(self._layout)
        self._target = LSPlotBokeh.show(plot, notebook_url=notebook_url, notebook_handle=True)

    # def html(self, filename=None):
    #     from bokeh.io import save, output_file
    #     if filename is None:
    #         raise ValueError("filename must be provided")
    #     output_file(filename)
    #     save(self._layout)

    
    def set_data(self, data, counts=200):
        super(LSPlotBokeh, self).set_data(data)
        self._counts = counts
        self._cold = []        
        ds = self._data
        if isinstance(ds, Dummy_g1data.Dummy_g1data):
            # from bokeh.palettes import Category10
            # import itertools
            # colors = itertools.cycle(Category10[10])
            dx = 1/ds._size
            x = np.linspace(0+dx/2,1-dx/2,ds._size)  # spaced x axis
            for i,_ in enumerate(ds.kinds,0):
                xy,_ = ds.gen_pt(x=x, kind=i)
                self._cold.append( LSPlotBokeh.ColumnDataSource(data=dict(x=xy[:,0],y=xy[:,1]))  )
                self._figure_gn.line('x','y',source=self._cold[i], line_width=3, line_alpha=0.6, color='red')
        if self._model is not None:
            self.update_ls()

    def update(self):
        if self._model is not None and self._data is not None:
            self.update_ls()
            LSPlotBokeh.push_notebook(handle=self._target)

    @without_document_lock
    def update_ls(self):
        model = self._model
        counts = self._counts
        ds   = self._data.ds_array.prefetch(counts).batch(counts).take(1)
        dc   = self._data[0:counts]
        dc._counts = counts
        ts,tl = ds.make_one_shot_iterator().get_next()
        def normalize(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        ## IS VAE
        if issubclass(type(model), models.base.VAE):
            if model.latent_dim == 2:
                m,v = model.encode(ts, training=False)
                z = model.reparametrize(m,v)
                v = tf.exp(0.5 * v) * 500.
                data=dict(  mx=m[:,0].numpy(), my=m[:,1].numpy(),
                            vx=v[:,0].numpy(), vy=v[:,1].numpy(), 
                            v_sum=(v[:,0].numpy()+v[:,1].numpy()),
                            zx=z[:,0].numpy(), zy=z[:,1].numpy(),
                            tcentro=normalize(dc['tcentro']),
                            label=tl.numpy()
                            )
                self._data_ls.data = data                
        
        ## IS GAN
        elif issubclass(type(model), models.base.GAN):
            if model.latent_dim == 2:
                self._figure_ls.x_range=LSPlotBokeh.Range1d(-5,5)
                self._figure_ls.y_range=LSPlotBokeh.Range1d(-5,5)
                # m = v = model.encode(ts, training=False)
                z = m = v = tf.random.normal(tf.shape(ts))
                data=dict(  mx=m[:,0].numpy(), my=m[:,1].numpy(),
                            vx=v[:,0].numpy(), vy=v[:,1].numpy(),
                            v_sum=(v[:,0].numpy()+v[:,1].numpy()),
                            zx=z[:,0].numpy(), zy=z[:,1].numpy(),
                            tcentro=normalize(dc['tcentro']),
                            label=tl.numpy()
                            )       
                self._data_ls.data = data
                


    def plot_generative(self, x, y):
        md = self._model
        XY = md.decode(tf.convert_to_tensor([[x,y]]), training=False, apply_sigmoid=True)
        X,Y = tf.split(XY[0], 2)
        data = dict( x=X.numpy(), y=Y.numpy() )
        self._data_gn.data = data

    def display_event(self, div, attributes=[], style = 'float:left;clear:left;font_size=10pt'):
        "Build a suitable CustomJS to display the current event in the div model."
        return LSPlotBokeh.CustomJS(args=dict(div=div, inx=self._inx), code="""
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





