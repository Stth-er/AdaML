import os
import odl
import torch
import pydicom
import numpy as np
from PIL import Image
from glob import glob
import SimpleITK as sitk
from odl.contrib import torch as odl_torch
import cv2

class CTSlice_Provider():
    def __init__(self, base_path=r"D:\Desktop", poission_level=5e6, gaussian_level=0.05, num_view=90):
        self.base_path = base_path
        self.slices_path = glob(os.path.join(self.base_path, '*/*.dcm'))
        self.poission_level = poission_level
        self.gaussian_level = gaussian_level
        self.num_view = num_view
        self.radon_full, self.iradon_full, self.fbp_full, self.op_norm_full = self._radon_transform(num_view=360)
        #self.radon_curr, self.iradon_curr, self.fbp_curr, self.op_norm_curr = self._radon_transform(num_view=num_view)
        self.radon_curr, self.iradon_curr, self.fbp_curr, self.op_norm_curr = self._radon_transform(num_view=self.num_view)
        

    def _radon_transform(self, num_view, start_ang=0, end_ang=2 * np.pi, num_detectors=800):
        # the function is used to generate fp, bp, fbp functions
        # the physical parameters is set as MetaInvNet and EPNet
        xx = 200
        space = odl.uniform_discr([-xx, -xx], [xx, xx], [256, 256] , dtype='float32')   #resize the gt, so change   [256, 256]  [512, 512] 
        angles = np.array(num_view).astype(int)
        angle_partition = odl.uniform_partition(start_ang, end_ang, angles)
        detector_partition = odl.uniform_partition(-480, 480, num_detectors)
        geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
        operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

        op_norm = odl.operator.power_method_opnorm(operator)
        op_norm = torch.from_numpy(np.array(op_norm * 2 * np.pi)).double().cuda()

        op_layer = odl_torch.operator.OperatorModule(operator)
        op_layer_adjoint = odl_torch.operator.OperatorModule(operator.adjoint)
        fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)
        op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

    def ril(self, num_view, start_ang=0, end_ang=2 * np.pi, num_detectors=800):
        # def ril(self, num_view=96, start_ang=-5/12*np.pi, end_ang=5/12*np.pi, num_detectors=800):
        xx = 200
        space = odl.uniform_discr([-xx, -xx], [xx, xx], [256, 256] , dtype='float32')  # [256, 256]   [512, 512]
        angles = np.array(num_view).astype(int)
        angle_partition = odl.uniform_partition(start_ang, end_ang, angles)
        detector_partition = odl.uniform_partition(-480, 480, num_detectors)
        geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
        operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

        op_norm = odl.operator.power_method_opnorm(operator)
        op_norm = torch.from_numpy(np.array(op_norm * 2 * np.pi)).double().cuda()

        op_layer = odl_torch.operator.OperatorModule(operator)
        op_layer_adjoint = odl_torch.operator.OperatorModule(operator.adjoint)
        fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)
        op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_adjoint, op_layer_fbp, op_norm


    def transform_read(self, path):
        try:
            ds = pydicom.dcmread(path)
        except AttributeError:
            ds = pydicom.read_file(path)
        dcm = ds
        dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        data_slice = dcm.image
        data_slice = np.array(data_slice).astype(float)
        # print(path)
        # print(np.max(data_slice), np.min(data_slice))

        #gt0 = sitk.GetImageFromArray(data_slice)
        #sitk.WriteImage(gt0, "./show/y0.nii.gz")
        
        #print('data_slice.shape:', data_slice.shape)
        data_slice = cv2.resize(data_slice, (256, 256), interpolation = cv2.INTER_CUBIC) # resize
        #print('resize data_slice.shape:', data_slice.shape)
        
        data_slice = (data_slice - np.min(data_slice)) / (np.max(data_slice) - np.min(data_slice))

        # the following code is used to generate projections with noise in the way of odl package
        phantom = torch.from_numpy(data_slice).unsqueeze(0).type(torch.FloatTensor)
        sino=self.radon_curr(phantom)
        #print('sino.shape:',sino.shape) #torch.Size([1, 90, 800])

        # the following part code is used to randomly choose sinograms to satisfy the sparse-view requeirement
        # add poission noise
        intensityI0=self.poission_level
        scale_value=torch.from_numpy(np.array(intensityI0).astype(float))
        normalized_sino=torch.exp(-sino/sino.max())
        th_data=np.random.poisson(scale_value*normalized_sino)
        sino_noisy=-torch.log(torch.from_numpy(th_data)/scale_value)
        sino_noisy = sino_noisy*sino.max()

        # add Gaussian noise
        noise_std=self.gaussian_level
        noise_std=np.array(noise_std).astype(float)
        nx,ny=np.array(self.num_view).astype(int),np.array(800).astype(int)
        noise = noise_std*np.random.randn(nx,ny)
        noise = torch.from_numpy(noise)
        sino_noisy = sino_noisy + noise

        # print(f"type(sino_noisy):{type(sino_noisy)}, sino_noisy.shape:{sino_noisy.shape}")  
        # #<class 'torch.Tensor'>, sino_noisy.shape:torch.Size([1, 90, 800])
        # if sino_noisy.is_cuda:
        #     sino_noisy = sino_noisy.cpu()
        # sino_noisy = sino_noisy.squeeze(0)
        # sino_noisy = sino_noisy.numpy()
        # print(f"type(sino_noisy):{type(sino_noisy)}, sino_noisy.shape:{sino_noisy.shape}")  
        # print(f'np.min(sino_noisy):{np.min(sino_noisy)}, np.max(sino_noisy):{np.max(sino_noisy)}')
        # img_np = (sino_noisy - np.min(sino_noisy)) / (np.max(sino_noisy) - np.min(sino_noisy)) * 255.0
        # img= img_np.astype(np.uint8)
        # image = Image.fromarray(img, mode='L')
        # destination_path = './show/iter_N_sinogram.png'
        # image.save(destination_path)
        # print(f"Saved: {destination_path}")


        fbp_u=self.fbp_curr(sino_noisy)
        phantom=phantom#.type(torch.DoubleTensor)
        fbp_u=fbp_u#.type(torch.DoubleTensor)
        sino_noisy=sino_noisy#.type(torch.DoubleTensor)

        return fbp_u, phantom   #FBP, target

if __name__ == '__main__':

    path =  '/data1/gdli/Dataset/Meta_data_N/meta_test_1/N/N005_1-12.dcm'  #"/data1/gdli/Dataset/Meta_data/meta_test_1/L/L273_1-081.dcm"
    trans = CTSlice_Provider(path)

    fbp_u, phantom = trans.transform_read(path)
    print('fbp_u.shape, phantom.shape:', fbp_u.shape, phantom.shape) #torch.Size([1, 256, 256]) torch.Size([1, 256, 256])
    # print(fbp_u, phantom)