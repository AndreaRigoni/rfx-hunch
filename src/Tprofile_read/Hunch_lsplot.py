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
        assert isinstance(data, Dummy_g1data.FiniteSequence1D), "Input variables should be FiniteSequence1D"
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
            self.plot_gn(x,y)
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

    def plot_gn(self, x, y):
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









"""
.########.##.....##....###....##.....##.########..##.......########..######.
.##........##...##....##.##...###...###.##.....##.##.......##.......##....##
.##.........##.##....##...##..####.####.##.....##.##.......##.......##......
.######......###....##.....##.##.###.##.########..##.......######....######.
.##.........##.##...#########.##.....##.##........##.......##.............##
.##........##...##..##.....##.##.....##.##........##.......##.......##....##
.########.##.....##.##.....##.##.....##.##........########.########..######.
"""

def ipv_linkdata_exameple():
    import ipyvolume as ipv
    from bokeh.io import output_notebook, show
    from bokeh.plotting import figure

    import ipyvolume.bokeh
    output_notebook()
    import vaex
    ds = vaex.example()
    N = 10000

    ipv.figure()
    quiver = ipv.quiver(ds.data.x[:N],  ds.data.y[:N],  ds.data.z[:N],
                    ds.data.vx[:N], ds.data.vy[:N], ds.data.vz[:N],
                    size=1, size_selected=5, color_selected="grey")
    ipv.xyzlim(-30, 30)
    ipv.show()    
    print('ipv done..')
    tools = "wheel_zoom,box_zoom,box_select,lasso_select,help,reset,"
    p = figure(title="E Lz space", tools=tools, width=500, height=500)
    r = p.circle(ds.data.Lz[:N], ds.data.E[:N],color="navy", alpha=0.2)
    # A 'trick' from ipyvolume to link the selection (one way traffic atm)
    ipyvolume.bokeh.link_data_source_selection_to_widget(r.data_source, quiver, 'selected')
    show(p)
    print('bok done..')


def bokeh_cb_example_1():
    """ Demonstration of how to register event callbacks using an adaptation
    of the color_scatter example from the bokeh gallery
    """
    import numpy as np
    from bokeh.io import show, output_file
    from bokeh.plotting import figure
    from bokeh import events
    from bokeh.models import CustomJS, Div, Button
    from bokeh.layouts import column, row

    def display_event(div, attributes=[], style = 'float:left;clear:left;font_size=10pt'):
        "Build a suitable CustomJS to display the current event in the div model."
        return CustomJS(args=dict(div=div), code="""
            var attrs = %s; var args = [];
            for (var i = 0; i<attrs.length; i++) {
                args.push(attrs[i] + '=' + Number(cb_obj[attrs[i]]).toFixed(2));
            }
            var line = "<span style=%r><b>" + cb_obj.event_name + "</b>(" + args.join(", ") + ")</span>\\n";
            var text = div.text.concat(line);
            var lines = text.split("\\n")
            if (lines.length > 35)
                lines.shift();
            div.text = lines.join("\\n");
        """ % (attributes, style))


    x = np.random.random(size=4000) * 100
    y = np.random.random(size=4000) * 100
    radii = np.random.random(size=4000) * 1.5
    colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)]

    p = figure(tools="pan,wheel_zoom,zoom_in,zoom_out,reset")
    p.scatter(x, y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)

    div = Div(width=400, height=p.plot_height, height_policy="fixed")
    button = Button(label="Button", button_type="success")
    layout = column(button, row(p, div))

    ## Events with no attributes
    button.js_on_event(events.ButtonClick, display_event(div)) # Button click
    p.js_on_event(events.LODStart, display_event(div))         # Start of LOD display
    p.js_on_event(events.LODEnd, display_event(div))           # End of LOD display

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

    for event in point_events:
        p.js_on_event(event, display_event(div, attributes=point_attributes))

    p.js_on_event(events.MouseWheel, display_event(div, attributes=wheel_attributes))
    p.js_on_event(events.Pan,        display_event(div, attributes=pan_attributes))
    p.js_on_event(events.Pinch,      display_event(div, attributes=pinch_attributes))

    output_file("js_events.html", title="JS Events Example")
    show(layout)




def bokeh_cb_example_2():
    from bokeh.layouts import column
    from bokeh.models import CustomJS, ColumnDataSource, Slider
    from bokeh.plotting import figure, output_file, show

    output_file("callback.html")

    x = [x*0.005 for x in range(0, 200)]
    y = x

    source = ColumnDataSource(data=dict(x=x, y=y))

    plot = figure(plot_width=400, plot_height=400)
    plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

    callback = CustomJS(args=dict(source=source), code="""
            var data = source.data;
            var f = cb_obj.value
            var x = data['x']
            var y = data['y']
            for (var i = 0; i < x.length; i++) {
                y[i] = Math.pow(x[i], f)
            }
            source.change.emit();
        """)

    slider = Slider(start=0.1, end=4, value=1, step=.1, title="power", callback=callback)
    layout = column(slider, plot)
    show(layout)





