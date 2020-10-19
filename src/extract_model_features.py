from networks.deeplabv3plus_2 import DeepLabV3Plus2

net = DeepLabV3Plus2('resnet101', num_classes=1)

state_dict = torch.load('models/DeepLabV3Plus2_ResNet50/VOC2012/pascal_voc/DeepLabV3Plus2_ResNet50_epoch-30.pth')
net.load_state_dict(state_dict)

def print_shape(self, inputs, outputs):
    print(outputs.shape)

net.backbone.layer3.register_forward_hook(print_shape)

db_train = DAVIS(seqs_key=train_dataset,
                 root_dir=db_root_dir,
                 crop_size=train_crop_size,
                 transform=composed_transforms,
                 multi_object=train_multi_object)

db_test = DAVIS(seqs_key=test_dataset,
                root_dir=db_root_dir,
                transform=tr.ToTensor())