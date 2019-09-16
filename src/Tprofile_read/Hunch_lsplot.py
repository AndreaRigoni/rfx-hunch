from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Hunch_utils as Htils

import Dummy_g1data
import Dataset_QSH

import numpy as np
import tensorflow as tf
import abc

import models
import models.AEFIT4
import models.AEFIT5
import models.Compose

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
        self._feed_data  = None
        self._browse_model = None
        
    @abc.abstractmethod        
    def set_model(self, model):
        # assert isinstance(model, AEFIT.Hunch_VAE), "Input variables should be AEFIT"
        self._model = model
        
    @abc.abstractmethod        
    def set_data(self, data):
        # assert isinstance(data, Dummy_g1data.FiniteSequence1D), "Input variables should be FiniteSequence1D"
        self._data = data    
            
    def train_browse_model(self, Model):
        model = self._model
        data  = self._data
                    
        # tot_len = len(data)
        # dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn, batch_size=FLAGS.batch_size))
        ds = data.ds_array.batch(100).map(lambda x,y: (model.encode(x, training=False)[0],y))
        self._browse_model = Model(latent_dim=2, feature_dim=model.latent_dim)
        # self._browse_model.inference_net.summary()
        # self._browse_model.generative_net.summary()
        # return [x for x in ds.map(lambda x,y: (x,x)).take(1)][0]
        models.base.train(self._browse_model, ds, epoch=5, batch=None)


class LSPlotBokeh(LSPlot):
    from bokeh.io import show, output_notebook, push_notebook
    from bokeh import events
    from bokeh.models import CustomJS, Div, Button, Slider, Toggle, Span
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

    

    def __init__(self, *args, **kwargs):
        super(LSPlotBokeh,self).__init__(*args,**kwargs)
        self._target = None
        self._doc = None

        self._figure_ls = LSPlotBokeh.figure(plot_width=800, plot_height=800,tools="save,pan,box_zoom,zoom_in,zoom_out,reset,crosshair")
        self._figure_gn = LSPlotBokeh.figure(plot_width=800, plot_height=800,tools="save,pan,zoom_in,zoom_out,reset",x_range=(0,1),y_range=(0,1))
        self._figure_Sabs = LSPlotBokeh.figure(plot_width=400, plot_height=200,tools="",x_range=(0,1),y_range=(0,1))
        self._figure_Sarg = LSPlotBokeh.figure(plot_width=400, plot_height=200,tools="",x_range=(0,1),y_range=(0,1))
        self._figure_ls.xaxis.major_label_text_font_size = "20pt"
        self._figure_ls.yaxis.major_label_text_font_size = "20pt"
        self._figure_gn.xaxis.major_label_text_font_size = "20pt"
        self._figure_gn.yaxis.major_label_text_font_size = "20pt"

        # self._div = LSPlotBokeh.Div(width=800, height=10, height_policy="fixed")        

        # trace mouse position
        self._inx = LSPlotBokeh.TextInput(value='', width=150)
        self._pos = LSPlotBokeh.ColumnDataSource(data=dict(x=[0],y=[0],dim=[0]))        
        def posx_cb(attr, old, new):
            pos = self._pos
            x,y = [float(x.strip()) for x in new.split(',')]
            pos.data['x'][0] = x
            pos.data['y'][0] = y
            self._doc.add_next_tick_callback(lambda: self.plot_generative(x,y))
            self._doc.add_next_tick_callback(lambda: self.plot_lsgen_point(x,y))            
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

        self._data_ls = LSPlotBokeh.ColumnDataSource(data=dict(selected_pt=[],mx=[],my=[],vx=[],vy=[],zx=[],zy=[],tcentro=[],tbordo=[],
                                                               label=[],Ip=[],dsxm=[],dens=[],F=[],TH=[],NS=[]
                                                               ) )
        self._figure_ls.scatter('zx','zy',name='sample', legend='sample', size=10, source=self._data_ls, color='grey', alpha=0.2, line_width=0)        
        # self._data_ls_selected_pt = LSPlotBokeh.ColumnDataSource( data=dict(x=[],y=[]) )        
        # def add_cross():            
        #     self._figure_ls.ray(x='x', y='y', source=self._data_ls_selected_pt, length=0, angle=0, line_width=2)
        #     self._figure_ls.ray(x='x', y='y', source=self._data_ls_selected_pt, length=0, angle=[np.pi/2,3*np.pi/2], line_width=2)
        # add_cross()

        def add_sl_glyph(name, field=None, mapper=self._mapper3):
            if field is None: field = name
            self._ls_glyphs += [self._figure_ls.circle('mx','my',name=name, legend=name,
                                                            size=10, 
                                                            source=self._data_ls, 
                                                            alpha=0.5, 
                                                            line_width=0,
                                                            fill_color={'field': field, 'transform': mapper}
                                                            )]
        add_sl_glyph('label', mapper=self._mapper4)
        add_sl_glyph('tcentro',)
        add_sl_glyph('tbordo',)
        add_sl_glyph('Ip',  )
        add_sl_glyph('dsxm',)
        add_sl_glyph('log dens', field='dens',)
        add_sl_glyph('F',   )
        add_sl_glyph('NS', )

        self._figure_ls.legend[0].visible = False
        toggle_ls_glyphs(None)
        for event in LSPlotBokeh.point_events:
            self._figure_ls.js_on_event(event, self.display_event(attributes=LSPlotBokeh.point_attributes))


        # NG PLOT        
        self._data_gn = LSPlotBokeh.ColumnDataSource(data=dict(x=[],y=[]))
        self._figure_gn.line('x','y', source=self._data_gn, color='black', line_width=3, line_dash=[10, 10])
        self._figure_gn.scatter('x','y', source=self._data_gn, size=10, color='blue' )

        # WIDGETS #
        self._b1 = LSPlotBokeh.Button(label="Update ls", button_type="success", width=150)
        self._b1.on_click(self.update_ls)
        self._b2 = LSPlotBokeh.Button(label="label", width=150)
        self._b2.on_click(lambda: toggle_ls_glyphs('label'))
        self._b3 = LSPlotBokeh.Button(label="Te center", width=150)
        self._b3.on_click(lambda: toggle_ls_glyphs('tcentro'))
        self._b4 = LSPlotBokeh.Button(label="Te bondary", width=150)
        self._b4.on_click(lambda: toggle_ls_glyphs('tbordo'))
        self._b5 = LSPlotBokeh.Button(label="Ip", width=150)
        self._b5.on_click(lambda: toggle_ls_glyphs('Ip'))
        self._b6 = LSPlotBokeh.Button(label="dsxm", width=150)
        self._b6.on_click(lambda: toggle_ls_glyphs('dsxm'))
        self._b7 = LSPlotBokeh.Button(label="log dens", width=150)
        self._b7.on_click(lambda: toggle_ls_glyphs('log dens'))
        self._b8 = LSPlotBokeh.Button(label="F", width=150)
        self._b8.on_click(lambda: toggle_ls_glyphs('F'))
        self._b9 = LSPlotBokeh.Button(label="NS", width=150)
        self._b9.on_click(lambda: toggle_ls_glyphs('NS'))

        self._layout = LSPlotBokeh.column( 
            LSPlotBokeh.row(self._figure_ls,self._figure_gn,
                LSPlotBokeh.column(
                    self._b1,
                    self._b2,
                    self._b3,
                    self._b4,
                    self._b5,
                    self._b6,
                    self._b7,
                    self._b8,
                    self._b9,
                    self._inx,
                )),
            #LSPlotBokeh.row(self._div)
            # LSPlotBokeh.column( self._figure_Sabs, self._figure_Sarg ),
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

    # def set_model(self, model):
    #     model_outputs_names = [ n. for n ]
    #     pass
    
    def set_data(self, data, feed_data=None, counts=200, show_kinds=True):
        super(LSPlotBokeh, self).set_data(data)
        self._counts = counts
        self._cold = []
        if feed_data is None: self._feed_data = self._data.ds_array
        else                : self._feed_data = feed_data
        ds = self._data
        if isinstance(ds, Dummy_g1data.Dummy_g1data) and show_kinds is True:
            # from bokeh.palettes import Category10
            # import itertools
            # colors = itertools.cycle(Category10[10])
            dx = 1/ds._size
            x = np.linspace(0+dx/2,1-dx/2,ds._size*10)  # spaced x axis
            for i,_ in enumerate(ds.kinds,0):
                xy,_ = ds.gen_pt(x=x, kind=i)
                self._cold.append( LSPlotBokeh.ColumnDataSource(data=dict(x=xy[:,0],y=xy[:,1]))  )
                self._figure_gn.line('x','y',source=self._cold[i], line_width=5, line_alpha=0.6, color=self._mapper4.palette[i] )
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
        ds   = self._feed_data.batch(counts).take(1)
        dc   = self._data[0:counts]
        # dc._counts = counts # to handle a bug that is going to be fixed soon
        
        
        ts,tl = [x for x in ds][0]
        def normalize(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))

        # if model.latent_dim > 2:
        #     from sklearn.manifold import TSNE
        #     tSNE_m = TSNE(n_components=2)
        #     tSNE_v = TSNE(n_components=2)
        #     for xy,_ in ds.ds_array.batch(counts).take(10):
        #         m,v = model.encode(xy)
        #         tSNE_m.fit(m)
        #         tSNE_v.fit(v)
        
        ## IS VAE
        if issubclass(type(model), models.Compose.Compose):
            tl = tl[0]
            
        if issubclass(type(model), models.base.VAE):
            m,v = model.encode(ts, training=False)
            z = model.reparametrize(m,v)
            v = tf.exp(0.5 * v) * 500.
            if self._browse_model:
                m,v = self._browse_model.encode(m)
                z   = self._browse_model.reparametrize(z)
                v = tf.exp(0.5 * v) * 500.
            elif model.latent_dim != 2:
                pass
            
        ## IS GAN
        elif issubclass(type(model), models.base.GAN):
            if model.latent_dim == 2:
                self._figure_ls.x_range=LSPlotBokeh.Range1d(-5,5)
                self._figure_ls.y_range=LSPlotBokeh.Range1d(-5,5)                                
                z = m = v = tf.random.normal(tf.shape(ts))

        if model.latent_dim == 2:                
            data=dict(
                        mx=m[:,0].numpy(), my=m[:,1].numpy(),
                        vx=v[:,0].numpy(), vy=v[:,1].numpy(), 
                        v_sum=(v[:,0].numpy()+v[:,1].numpy()),
                        zx=z[:,0].numpy(), zy=z[:,1].numpy(),
                        tcentro=normalize(dc['tcentro']),
                        tbordo=normalize(dc['tbordo']),
                        Ip=normalize(dc['Ip']),
                        dsxm=normalize(dc['Te_dsxm']),
                        dens=normalize(dc['dens']),
                        F=normalize(dc['F']),
                        NS=normalize(dc['NS']),
                        label=tl.numpy()
                    )
            self._data_ls.data = data
        

    def plot_generative(self, x, y, lasso_list=None):
        md = self._model
        XY = md.decode(tf.convert_to_tensor([[x,y]]), training=False)
        if isinstance(XY, list): XY = XY[0]   # if list of outputs take first one
        X,Y = tf.split(XY[0], 2)
        data = dict( x=X.numpy(), y=Y.numpy() )
        self._data_gn.data = data

    def plot_lsgen_point(self, x,y):
        try: 
            pt = self._figure_ls.select_one({'name': 'selected_pt'})
            pt.glyph.x = x
            pt.glyph.y = y
        except:
            pt = self._figure_ls.cross(x,y, name='selected_pt', color='black', size=30, line_width=4)
            self._figure_ls.renderers.extend([pt])
        pass

    def plot_spectrum(self, x, y, lasso_list=None):
        ''' plot reconstructed spectrum input '''
        pass


    def display_event(self, attributes=[], style = 'float:left;clear:left;font_size=10pt'):
        "Build a suitable CustomJS to display the current event in the div model."
        return LSPlotBokeh.CustomJS(args=dict(inx=self._inx), code="""
            var attrs = %s; var args = [];
            for (var i = 0; i<attrs.length; i++)
                args.push(attrs[i] + '=' + Number(cb_obj[attrs[i]]).toFixed(2));
            var line = ""
            var x = cb_obj[attrs[0]]
            var y = cb_obj[attrs[1]]
            if (cb_obj.event_name == "tap")
                inx.value = Number(x).toFixed(3) + "," + Number(y).toFixed(3);
            // line += "<span style=%r><b>" + cb_obj.event_name + "</b>(" + Number(x).toFixed(3) + ","
            //                                                               + Number(y).toFixed(3) + 
            //                                                               ")</span>\\n";
            // var text = div.text.concat(line);
            // var lines = text.split("\\n")
            // if (lines.length > 15)
            //     lines.shift();
            // div.text = lines.join("\\n");
        """ % (attributes, style))





