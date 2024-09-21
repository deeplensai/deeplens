import yaml
import torch.nn as nn
from deeplens.backbones.backbones import ConvBlock
from deeplens.utils.config_reader import ConfigReader


MODULES = {
    'ConvBlock': ConvBlock,
    'torch.nn': nn
}

class ModelBuilderYaml:
    def __init__(self, config_file):
        self.config = ConfigReader(config_file)
        self.model = self._build_model()

    def _build_model(self):
        model = nn.Sequential()
        for layer in self.config.get('layers'):
            layer_type = layer['type']

            for module in MODULES:
                layer_class = getattr(module, type, None)
                if layer_class is not None:
                    break
            if layer_class is None:
                raise ValueError(f"Layer type {layer_type} not found in torch.nn")
            
            model.add_module(layer['name'], layer_class(**layer['args']))
        return model

    def get_model(self):
        return self.model
    

if __name__ == "__main__":
    config_file = '/workspaces/deeplens/deeplens/config/models/lenet5.yaml'
    model_builder = ModelBuilderYaml(config_file)
    model = model_builder.get_model()
    print(model)