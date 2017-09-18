from IPython.display import clear_output, Image, display, HTML
import os
import tensorflow as tf
import numpy as np


def w2color(w, w_range=(-1.0,1.0), color_scale = ['#d6604d', '#f4a582', '#fddbc7', '#f7f7f7','#d1e5f0', '#92c5de', '#4393c3']):

    #linear mapping
    #[-1, 1] --> [0, max]
    #max_i = float(len(color_scale) - 1)
    #a = (w_range[1] - w_range[2])/max_i
    #w_map = (w - w_range[0])*a + max_i
    #return color_scale[round(w_map)]
    
    color_bin = np.digitize(w, np.linspace(w_range[0], w_range[1], len(color_scale)))
    
    return color_scale[color_bin - 1]
    
def make_colored_text(sample_text, states, cell_id, layer_id=0):
    
    if (len(states) != len(sample_text)):
        raise Exception("Invalid.")
    
    out_html = '<span style="white-space: per-line; font-family: Courier, Monaco, mono space">'
    out_html += '{}</span>'
    
    
    span_html_t = '<span style="background-color:{}">{}</span>' 
    
    #TODO: make this vectorized 
    span_html_arr = []
    for i, c in enumerate(sample_text):
        w = np.tanh(states[i][layer_id].c[0][cell_id])
        span_html = span_html_t.format(w2color(w),c)
        span_html_arr.append(span_html)
    
    
    out_text = ''.join(span_html_arr)
    
    out_html=out_html.format(out_text)
    
    return out_html.replace('\n', '<br>')
    
    
def save_lstm_vis(filename, samp, states):
    
    num_layers = len(states[0])
    num_memory_cells = int(states[0][0].c.shape[1])
    
    print("Number of layers: {}".format(num_layers))
    print("Number of memory cells (LSTM size): {}".format(num_memory_cells))
    
    for layer_id in range(num_layers):
        filename_out = "{}_{}.html".format(filename, layer_id)
        print("Saving {}...".format(filename_out))
        
        with open(filename_out, 'w') as f_out:
            for cell_id in range(num_memory_cells):
                f_out.write("[{}] ".format(cell_id))
                f_out.write(make_colored_text(samp, states, cell_id=cell_id, layer_id=layer_id))
                f_out.write("<hr>")

                
def test_colors():
    
    out_html = '<span style="white-space: per-line; font-family: Courier, Monaco, mono space"'
    out_html += '{}</span>'
    
    span_html = '<span style="background-color:{html_color}">{char}</span>'
    
    out_text = ''.join([ span_html.format(html_color=w2color(w), char=str(w))\
                        for w in np.random.uniform(low=-1.0, high=1.0, size=50)])
    
    out_html=out_html.format(out_text)
    
    return out_html


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def get_arr_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    # Get the number of characters per batch and number of batches we can make
    characters_per_batch = n_seqs * n_steps
    n_batches = len(arr)//characters_per_batch
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * characters_per_batch]
    
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:, n:n+n_steps]
        # The targets, shifted by one
        y = np.zeros_like(x)
        
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], np.roll(arr[:, 0], -1)
        
        yield x, y


# Show tensorboard inside Jupyter notebook
# https://gist.github.com/Algomancer/108908849e30083121dcdce380a9114f
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))