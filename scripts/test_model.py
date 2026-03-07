from torchvision.models import resnet50, ResNet50_Weights
print('downloading...')
m = resnet50(weights=ResNet50_Weights.DEFAULT)
print('done')
