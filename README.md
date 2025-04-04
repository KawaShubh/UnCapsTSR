# UnCapsTSR: An Unsupervised Transformer-based Image Super-Resolution Approach for Capsule Endoscopy Images

The repository contains the official code for the work **"UnCapsTSR: An Unsupervised Transformer-based Image Super-Resolution Approach for Capsule Endoscopy Images"** 

**- Pre-Trained models**

The pretrained models can be downloaded from the drive link below
[📁 **Google Drive**](https://drive.google.com/drive/folders/1iSSbL5Nvz7f749N8NgjJ2dg3YerxJmL7?usp=sharing)

**- Training the model**

Training code has been released. To train the network, run the following command.
```javascript
python train.py -opt path_for_training_json_file
```
Note the following changes are needed to run the code.
- Change the root folder and training dataset path into train_ntireEx.json file located at options/train folder.

**- Testing the model**

To test your/our pre-trained model, you need to set root directory and dataset directory into `options/test/test_ntire_ex.json` file. Then run the following command to start the training.
```javascript
python test.py -opt PATH-to-json_file

```

**- Requirement of packages**

The list of packages required to run the code is given in `uncapstsr.yml` file.

For any problem, you may contact at <anjali.sarvaiya.as@gmail.com>.
