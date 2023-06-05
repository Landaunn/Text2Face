

def normalise_data(w, w_mean, w_std):
    device = w.device
    w = w - w_mean.to(device)
    w = w / w_std.to(device)
    return w

def denormalise_data(w, w_mean, w_std):
    device = w.device
    w = w * w_std.to(device)
    w = w + w_mean.to(device)
    return w