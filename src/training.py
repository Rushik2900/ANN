from src.utils.common import read_config
import argparse
from src.utils.data_mngmnt import get_data
from src.utils.model import create_model


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


    

if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--config", "c", default="config.yml")
    parsed_arg = args.parse_args()

    training(config_path= parsed_arg.config)