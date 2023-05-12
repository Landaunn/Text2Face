from pathlib import Path

def load_sg(network_pkl, device):
    import sys
    code_folder = Path('./').parent
    sg3_path = str(code_folder/"stylegan2")       
    sys.path.append(sg3_path)
    
    import dnnlib
    import legacy

    with dnnlib.util.open_url(network_pkl) as f:
        model = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    return model
    
