import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import math

from retinanet.anchors import Anchors
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet import losses
from retinanet.quantize import QConv2d

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def qconv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 quantized convolution with padding"""
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
            padding=dilation, groups=groups, bias=False, dilation=dilation)

def qconv1x1(in_planes, out_planes, stride=1):
    """1x1 quantized convolution"""
    return QConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class QBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = qconv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = qconv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = qconv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample # assumed to be low precision
        self.stride = stride

    def forward(self, x, num_bits, num_grad_bits):
        identity = x

        out = self.conv1(x, num_bits, num_grad_bits)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, num_bits, num_grad_bits)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, num_bits, num_grad_bits)
        out = self.bn3(out)

        if self.downsample is not None:
            for bl in self.downsample:
                if isinstance(bl, QConv2d):
                    identity = bl(identity, num_bits, num_grad_bits)
                else:
                    identity = bl(identity)

        out += identity
        out = self.relu(out)
        return out

class QBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
            base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = qconv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = qconv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample # assume this is a low precision module
        self.stride = stride

    def forward(self, x, num_bits, num_grad_bits):
        identity = x

        out = self.conv1(x, num_bits, num_grad_bits)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, num_bits, num_grad_bits)
        out = self.bn2(out)

        if self.downsample is not None:
            for bl in self.downsample:
                if isinstance(bl, QConv2d):
                    identity = bl(identity, num_bits, num_grad_bits)
                else:
                    identity = bl(identity)

        out += identity
        out = self.relu(out)

        return out


class QPyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(QPyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = QConv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest') # up samples are full precision
        self.P5_2 = QConv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = QConv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = QConv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = QConv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = QConv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = QConv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = QConv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs, num_bits, num_grad_bits):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5, num_bits, num_grad_bits)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x, num_bits, num_grad_bits)

        P4_x = self.P4_1(C4, num_bits, num_grad_bits)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x, num_bits, num_grad_bits)

        P3_x = self.P3_1(C3, num_bits, num_grad_bits)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x, num_bits, num_grad_bits)

        P6_x = self.P6(C5, num_bits, num_grad_bits)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x, num_bits, num_grad_bits)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class QRegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(QRegressionModel, self).__init__()

        self.conv1 = QConv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = QConv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = QConv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = QConv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = QConv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x, num_bits, num_grad_bits):
        out = self.conv1(x, num_bits, num_grad_bits)
        out = self.act1(out)

        out = self.conv2(out, num_bits, num_grad_bits)
        out = self.act2(out)

        out = self.conv3(out, num_bits, num_grad_bits)
        out = self.act3(out)

        out = self.conv4(out, num_bits, num_grad_bits)
        out = self.act4(out)

        out = self.output(out, num_bits, num_grad_bits)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class QClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(QClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = QConv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = QConv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = QConv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = QConv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = QConv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x, num_bits, num_grad_bits):
        out = self.conv1(x, num_bits, num_grad_bits)
        out = self.act1(out) # activations performed in full precision

        out = self.conv2(out, num_bits, num_grad_bits)
        out = self.act2(out)

        out = self.conv3(out, num_bits, num_grad_bits)
        out = self.act3(out)

        out = self.conv4(out, num_bits, num_grad_bits)
        out = self.act4(out)

        out = self.output(out, num_bits, num_grad_bits)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes * n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class QResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super().__init__()
        self.conv1 = QConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == QBasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == QBottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = QPyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = QRegressionModel(256)
        self.classificationModel = QClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, QConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs, num_bits, num_grad_bits):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch, num_bits, num_grad_bits)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = x
        for bl in self.layer1:
            x1 = bl(x1, num_bits, num_grad_bits)

        x2 = x1
        for bl in self.layer2:
            x2 = bl(x2, num_bits, num_grad_bits)

        x3 = x2
        for bl in self.layer3:
            x3 = bl(x3, num_bits, num_grad_bits)

        x4 = x3
        for bl in self.layer4:
            x4 = bl(x4, num_bits, num_grad_bits)

        features = self.fpn([x2, x3, x4], num_bits, num_grad_bits)

        regression = torch.cat([self.regressionModel(feature, num_bits, num_grad_bits) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature, num_bits, num_grad_bits) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
            

def qresnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model with quantized training.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = QResNet(num_classes, QBasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


# def resnet34(num_classes, pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(num_classes, QBasicBlock, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
#     return model


# def resnet50(num_classes, pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
#     return model


def qresnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model with quantized training.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = QResNet(num_classes, QBottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


# def resnet152(num_classes, pretrained=False, **kwargs):
#     """Constructs a ResNet-152 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
#     return model
