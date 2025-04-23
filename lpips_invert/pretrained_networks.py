from collections import namedtuple
import torch
from torchvision import models as tv
import torch
import torch.nn as nn


# from torchvision_abd import models as tv
# from torchvision_abd.models import as tv
class squeezenet_f(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet_f, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.flip = Flip()

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        # for x in range(2):
        #     self.slice1.add_module(str(x), pretrained_features[x])
        # for x in range(2, 5):
        #     self.slice2.add_module(str(x), pretrained_features[x])
        # for x in range(5, 8):
        #     self.slice3.add_module(str(x), pretrained_features[x])
        # for x in range(8, 10):
        #     self.slice4.add_module(str(x), pretrained_features[x])
        # for x in range(10, 11):
        #     self.slice5.add_module(str(x), pretrained_features[x])
        # for x in range(11, 12):
        #     self.slice6.add_module(str(x), pretrained_features[x])
        # for x in range(12, 13):
        #     self.slice7.add_module(str(x), pretrained_features[x])

        x = 0
        y = 0
        self.slice1.add_module(str(y), pretrained_features[x])
        y = y + 1
        x = x + 1 #0

        self.slice1.add_module(str(y), self.flip)
        y=y+1




        self.slice1.add_module(str(y), pretrained_features[x])
        y = y + 1
        x = x + 1 #1

        self.slice2.add_module(str(y), pretrained_features[x])
        y = y + 1
        x = x + 1 #2




        #need to edit the Fire object
        # fire_obj=Fire_invert(pretrained_features[x].inplanes, pretrained_features[x].squeeze.out_channels, pretrained_features[x].expand1x1.out_channels,
        #                      pretrained_features[x].expand3x3.out_channels)
        # fire_obj.squeeze=pretrained_features[x].squeeze
        # fire_obj.expand1x1=pretrained_features[x].expand1x1
        # fire_obj.expand3x3=pretrained_features[x].expand3x3

        fire_obj=int_Fire_invert(pretrained_features[x])
        self.slice2.add_module(str(y), fire_obj)
        y = y + 1
        x = x + 1 #3

        fire_obj=int_Fire_invert(pretrained_features[x])
        self.slice2.add_module(str(y), fire_obj)
        y = y + 1
        x = x + 1 #4

        self.slice3.add_module(str(y), pretrained_features[x])
        y = y + 1
        x = x + 1  # 5
        fire_obj=int_Fire_invert(pretrained_features[x])
        self.slice3.add_module(str(y), fire_obj)
        y = y + 1
        x = x + 1  # 6

        fire_obj=int_Fire_invert(pretrained_features[x])
        self.slice3.add_module(str(y), fire_obj)
        y = y + 1
        x = x + 1  # 7

        self.slice4.add_module(str(y), pretrained_features[x])
        y = y + 1
        x = x + 1  # 8

        fire_obj=int_Fire_invert(pretrained_features[x])
        self.slice4.add_module(str(y), fire_obj)
        y = y + 1
        x = x + 1  # 9

        fire_obj=int_Fire_invert(pretrained_features[x])
        self.slice5.add_module(str(y), fire_obj)
        y = y + 1
        x = x + 1  # 10

        fire_obj=int_Fire_invert(pretrained_features[x])
        self.slice6.add_module(str(y), fire_obj)
        y = y + 1
        x = x + 1  # 11

        fire_obj=int_Fire_invert(pretrained_features[x])
        self.slice7.add_module(str(y), fire_obj)
        y = y + 1
        x = x + 1  # 12















        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple("SqueezeOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7'])
        # vgg_outputs = namedtuple("SqueezeOutputs", ['relu1'])
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)
        # out = vgg_outputs(h_relu1)

        return out

class squeezenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.flip = Flip()

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h

        vgg_outputs = namedtuple("SqueezeOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7'])
        # vgg_outputs = namedtuple("SqueezeOutputs", ['relu1'])
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)
        # out = vgg_outputs(h_relu1)
        return out


class Flip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,X):
        return -X


class Fire_invert(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flip=True

        if(flip):
            x = self.squeeze_activation(-self.squeeze(x))
            return torch.cat(
                [self.expand1x1_activation(-self.expand1x1(x)), self.expand3x3_activation(-self.expand3x3(x))], 1
            )
        else:
            x = self.squeeze_activation(self.squeeze(x))
            return torch.cat(
                [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
            )


def int_Fire_invert(fire_org):
    fire_obj = Fire_invert(fire_org.inplanes, fire_org.squeeze.out_channels,
                           fire_org.expand1x1.out_channels,
                           fire_org.expand3x3.out_channels)
    fire_obj.squeeze = fire_org.squeeze
    fire_obj.expand1x1 = fire_org.expand1x1
    fire_obj.expand3x3 = fire_org.expand3x3


    return fire_obj

class alexnet_f(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet_f, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features

        # self.model = tv.AlexNet()
        self.flip=Flip()
        # weights = torch.load(
        #     'C:\\ML\\Second_PhD_Part\\Face_Anomaly_Appraisal\\pretrained_models\\alexnet-owt-7be5be79.pth')
        #
        # self.model.load_state_dict(weights)
        # alexnet_pretrained_features=self.model.features
        # alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        # alexnet_pretrained_features = Alexnet_inv(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5

        # for x in range(2):
        x=0
        y=0
        self.slice1.add_module(str(y), alexnet_pretrained_features[x])
        y=y+1
        x=x+1
        self.slice1.add_module(str(y), self.flip)
        y=y+1
        self.slice1.add_module(str(y), alexnet_pretrained_features[x])
        y=y+1
        x=x+1

        # for x in range(2, 5):
        self.slice2.add_module(str(y), alexnet_pretrained_features[x])
        y=y+1
        x=x+1

        self.slice2.add_module(str(y), alexnet_pretrained_features[x])
        y=y+1
        x=x+1

        self.slice2.add_module(str(y), self.flip)
        y=y+1
        self.slice2.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1

        # for x in range(5, 8):
        # self.slice3.add_module(str(x), alexnet_pretrained_features[x])

        self.slice3.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1

        self.slice3.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1

        self.slice3.add_module(str(y), self.flip)
        y = y + 1

        self.slice3.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1


        # for x in range(8, 10):
        #     self.slice4.add_module(str(x), alexnet_pretrained_features[x])

        self.slice4.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1
        self.slice4.add_module(str(y), self.flip)
        y = y + 1
        self.slice4.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1



        # for x in range(10, 12):
        #     self.slice5.add_module(str(x), alexnet_pretrained_features[x])


        self.slice5.add_module(str(y), alexnet_pretrained_features[x])
        y=y+1
        x=x+1
        self.slice5.add_module(str(y), self.flip)
        y=y+1
        self.slice5.add_module(str(y), alexnet_pretrained_features[x])
        y=y+1
        x=x+1



        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):

        # y=self.model(X)



        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out

class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features

        # self.model = tv.AlexNet()
        self.flip = Flip()
        # weights = torch.load(
        #     'C:\\ML\\Second_PhD_Part\\Face_Anomaly_Appraisal\\pretrained_models\\alexnet-owt-7be5be79.pth')
        #
        # self.model.load_state_dict(weights)
        # alexnet_pretrained_features=self.model.features
        # alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        # alexnet_pretrained_features = Alexnet_inv(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5

        # for x in range(2):
        x = 0
        y = 0
        self.slice1.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1
        # self.slice1.add_module(str(y), self.flip)
        # y = y + 1
        self.slice1.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1

        # for x in range(2, 5):
        self.slice2.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1

        self.slice2.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1

        # self.slice2.add_module(str(y), self.flip)
        # y = y + 1
        self.slice2.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1

        # for x in range(5, 8):
        # self.slice3.add_module(str(x), alexnet_pretrained_features[x])

        self.slice3.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1

        self.slice3.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1

        # self.slice3.add_module(str(y), self.flip)
        # y = y + 1

        self.slice3.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1

        # for x in range(8, 10):
        #     self.slice4.add_module(str(x), alexnet_pretrained_features[x])

        self.slice4.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1
        # self.slice4.add_module(str(y), self.flip)
        # y = y + 1
        self.slice4.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1

        # for x in range(10, 12):
        #     self.slice5.add_module(str(x), alexnet_pretrained_features[x])

        self.slice5.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1
        # self.slice5.add_module(str(y), self.flip)
        # y = y + 1
        self.slice5.add_module(str(y), alexnet_pretrained_features[x])
        y = y + 1
        x = x + 1

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):

        # y=self.model(X)

        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out



class vgg16_f(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16_f, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.flip = Flip()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        # for x in range(4):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(4, 9):
        #     self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(9, 16):
        #     self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(16, 23):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(23, 30):
        #     self.slice5.add_module(str(x), vgg_pretrained_features[x])

            # for x in range(2):
        x = 0
        y = 0
        self.slice1.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  #0
        self.slice1.add_module(str(y), self.flip)
        y = y + 1
        self.slice1.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1

        # for x in range(2, 5):
        self.slice1.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  #2

        self.slice1.add_module(str(y), self.flip)
        y = y + 1

        self.slice1.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1   #0-3


        self.slice2.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1

        self.slice2.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  #5

        self.slice2.add_module(str(y), self.flip)
        y = y + 1
        self.slice2.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1


        self.slice2.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  #7


        self.slice2.add_module(str(y), self.flip)
        y = y + 1


        self.slice2.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #4-8



        self.slice3.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1




        self.slice3.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  #10
        self.slice3.add_module(str(y), self.flip)
        y = y + 1

        self.slice3.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1



        self.slice3.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #12


        self.slice3.add_module(str(y), self.flip)
        y = y + 1

        self.slice3.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1



        self.slice3.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #14


        self.slice3.add_module(str(y), self.flip)
        y = y + 1


        self.slice3.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  # 15



        self.slice4.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1

        self.slice4.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #17


        self.slice4.add_module(str(y), self.flip)
        y = y + 1

        self.slice4.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  # 18



        self.slice4.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  #19


        self.slice4.add_module(str(y), self.flip)
        y = y + 1

        self.slice4.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  # 20





        self.slice4.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #21


        self.slice4.add_module(str(y), self.flip)
        y = y + 1

        self.slice4.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  # 22



        self.slice5.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1


        self.slice5.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #24


        self.slice5.add_module(str(y), self.flip)
        y = y + 1

        self.slice5.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  # 25



        self.slice5.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #26

        self.slice5.add_module(str(y), self.flip)
        y = y + 1

        self.slice5.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  # 27




        self.slice5.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #28

        self.slice5.add_module(str(y), self.flip)
        y = y + 1

        self.slice5.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  # 29



        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out



class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.flip = Flip()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        # for x in range(4):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(4, 9):
        #     self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(9, 16):
        #     self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(16, 23):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(23, 30):
        #     self.slice5.add_module(str(x), vgg_pretrained_features[x])

            # for x in range(2):
        x = 0
        y = 0
        self.slice1.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  #0

        self.slice1.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1

        # for x in range(2, 5):
        self.slice1.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  #2


        self.slice1.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1   #0-3


        self.slice2.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1

        self.slice2.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  #5


        self.slice2.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1


        self.slice2.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  #7




        self.slice2.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #4-8



        self.slice3.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1




        self.slice3.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  #10


        self.slice3.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1



        self.slice3.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #12




        self.slice3.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1



        self.slice3.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #14




        self.slice3.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  # 15



        self.slice4.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1

        self.slice4.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #17



        self.slice4.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  # 18



        self.slice4.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  #19




        self.slice4.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  # 20





        self.slice4.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #21




        self.slice4.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  # 22



        self.slice5.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1


        self.slice5.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #24




        self.slice5.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  # 25



        self.slice5.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #26



        self.slice5.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  # 27




        self.slice5.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1 #28



        self.slice5.add_module(str(y), vgg_pretrained_features[x])
        y = y + 1
        x = x + 1  # 29



        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class resnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if (num == 18):
            self.net = tv.resnet18(pretrained=pretrained)
        elif (num == 34):
            self.net = tv.resnet34(pretrained=pretrained)
        elif (num == 50):
            self.net = tv.resnet50(pretrained=pretrained)
        elif (num == 101):
            self.net = tv.resnet101(pretrained=pretrained)
        elif (num == 152):
            self.net = tv.resnet152(pretrained=pretrained)
        self.N_slices = 5

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h

        outputs = namedtuple("Outputs", ['relu1', 'conv2', 'conv3', 'conv4', 'conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)

        return out
