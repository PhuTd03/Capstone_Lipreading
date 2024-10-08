import os
import sys

from models import model_identifier

def local_directory(name, model_cfg, save_dir, output_dir):
    model_name = model_identifier(model_cfg)
    # local_path = model_name # lipreading_d0.2_ks3_nl5

    if save_dir is None:
        save_dir = os.getcwd()
    if not (name is None or name == ""):
        output_dir = os.path.join(save_dir, 'exp', name, model_name, output_dir)
    else:
        output_dir = os.path.join(save_dir, 'exp', model_name, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.chmod(output_dir, 0o775)
    print('local directory:', output_dir)
    return model_name, output_dir

    
