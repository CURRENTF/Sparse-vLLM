class Context:
    def __init__(self):
        self.is_prefill = False
        self.is_long_text = False
        self.cu_seqlens_q = None
        self.now_layer_idx = 0
        self.cache_manager = None
        self.sparse_controller = None
        self.sparse_config = None
        self.seqs = None
        self.decode_mid_o = None
        self.decode_mid_o_logexpsum = None
        self.decode_cuda_graph_static = False


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cache_manager=None, is_long_text=False, seqs=None):
    global _CONTEXT
    _CONTEXT.is_prefill = is_prefill
    _CONTEXT.is_long_text = is_long_text
    _CONTEXT.cu_seqlens_q = cu_seqlens_q
    _CONTEXT.now_layer_idx = 0
    _CONTEXT.cache_manager = cache_manager
    _CONTEXT.seqs = seqs

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
