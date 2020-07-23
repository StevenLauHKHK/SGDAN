# Self-Guided Dual Attention Network for UDC Image Restoration

# Requirements
* Python 3.5
* PyTorch 1.1.0
* CUDA 10.2
* Scipy 1.1.0
* PIL 5.2.0
* Opencv 3.4.2

# Image Patch Pairs Generation
* Change "gt_folder", "input_folder", "output_gt_folder", "output_input_folder" folder path in img_patch_gen.py
* Change "patch_size" and "stride" in img_patch_gen.py
* Run img_patch_gen.py

# Train
* Change the parameters
* Run pyhton train.py

# Test
Track 2
* python test.py --valid_file (test mat file) --saving_img_root (output png file root) --saving_mat_dir (result mat zip saving root) --load_name (model name)

# Pretrain model
* https://drive.google.com/drive/folders/1EAAycawumKEVaZVI2JoRX32tC1Er13Ar?usp=sharing
* https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_ad_cityu_edu_hk/EqXe_Rk9aFNKvwrBOOmToWkBv-JjCpbvWaDPZyiJ5TC24w?e=8FZcTl
