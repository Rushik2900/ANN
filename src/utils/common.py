import yml

def read_config(config_path):
    with open(config_path) as config_file:
        content = yml.safe_lode(config_file)
    
    return content