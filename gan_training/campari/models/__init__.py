from torch import nn
from gan_training.campari.models import generator


generator_dict = {
    'default': generator.Generator
}


class Nerf(nn.Module):
    def __init__(self, device="cuda", generator=None, discriminator=None, generator_test=None):
        super().__init__()
        self.device = device
        
        if generator is not None:
            self.generator = generator.to(device)
        else:
            self.generator = None

        if discriminator is not None:
            self.discriminator = discriminator.to(device)
        else:
            self.discriminator = None
        
        if generator_test is not None:
            self.generator_test = generator_test.to(device)
        else:
            self.generator_test = None
    
    def get_test_generator(self):
        if self.generator_test is not None:
            return self.generator_test
        else:
            return self.generator

    def forward(self, x):
        return self.generator(x)

    def get_generator_parameters(self):
        return self.generator.get_parameters()
