def model_identifier(model_cfg):
    return "lipreading_d{}_ks{}_nl{}".format(
        model_cfg.tcn_dropout, model_cfg.kernel_size, model_cfg.num_layers
    )