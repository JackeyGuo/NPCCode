"""
this file is the models factory.
change it for add models.
"""

from models import dense_dilated_aspp_network,unet_network,deeplab_like_network,CNN,\
    deeplab_dense_aspp,resnet_aspp,xception_aspp,deform_xception_aspp,resnet_aspp_5fold,VNet,CNN_v2,VNet_new

modelMap = dict()

modelMap["denseDilatedASPP"] = dense_dilated_aspp_network.DenseDilatedASPP
modelMap["unet"] = unet_network.Unet
modelMap["deeplab"] = deeplab_like_network.Deeplab
modelMap["cnn_v2"] = CNN_v2.cnn
#modelMap["unet_3d"] = unet_3d.Unet3D
modelMap["cnn_v1"] = CNN.cnn
modelMap["mynet"] = deeplab_dense_aspp.deeplab_dense_aspp
modelMap["resnet_aspp"] = resnet_aspp.resnet_aspp
modelMap["resnet_aspp_5fold"] = resnet_aspp_5fold.resnet_aspp
modelMap["xception_aspp"] = xception_aspp.xception_aspp
modelMap["deform_xception_aspp"] = deform_xception_aspp.deform_xception_aspp
modelMap["vnet"] = VNet.V_Net
modelMap["vnet_new"] = VNet_new.V_Net
#modelMap["mynet_dst"] = deeplab_dense_aspp.deeplab_dense_aspp
#modelMap["mynet_focal"] = deeplab_dense_aspp.deeplab_dense_aspp
