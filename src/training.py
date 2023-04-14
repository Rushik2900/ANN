from src.utils.common import read_config
import argparse
from src.utils.data_mngmnt import get_data
from src.utils.model import create_model, save_model
from src.utils.plot import save_plot
import os
import pandas as pd

def training(config_path):
    config = read_config(config_path)
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train, X_test, y_test, X_valid, y_valid)=get_data(validation_datasize)

    input_shape = config["params"]["input_shape"]
    no_classes = config["params"]["no_classes"]
    loss_function = config["params"]["loss_function"]
    optimizer = config["params"]["optimizer"]
    metrics = config["params"]["metrics"]
    
    model = create_model(input_shape, no_classes, loss_function, optimizer, metrics)

    EPOCH = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)

    history = model.fit(X_train, y_train, epochs=EPOCH, validation_data=VALIDATION_SET)

    model_name=config['artifacts']['model_name']
    artifacts_dir=config['artifacts']['artifacts_dir']
    model_dir=config['artifacts']['model_dir']
    model_dir_path=os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    save_model(model, model_name, model_dir_path)
    
    df=pd.DataFrame(history.history).plot(figsize=(10,7))
    plot_name=config['artifacts']['plot_name']
    plot_dir=config['artifacts']['plot_dir']
    plot_dir_path=os.path.join(artifacts_dir, plot_dir)
    os.makedirs(plot_dir_path, exist_ok=True)
    
    save_plot(df, plot_name, plot_dir_path)
    

if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_arg = args.parse_args()

    training(config_path= parsed_arg.config)
