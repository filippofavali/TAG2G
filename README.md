# Full pipeline commands 

1. Preprocess data and train VQVAE
2. Data featurization (gesture, audio, text)
3. Train diffusion model
4. Inference time diffusion model 


# Install venv requirements and packages 

Step #1: Create a new venv and run pip installer to install all required packages

    conda crate --name myenv python=3.11 
    conda activate myenv
    pip install -r requirements.txt

Step #2: Install PASE+ to encode audio 

    pip install git+https://github.com/santi-pdp/pase.git

Step #3 :Install PyTorch QRNN from github repo without cloning the repo locally (in order to use PASE+)
        
    pip install git+https://github.com/salesforce/pytorch-qrnn
    
Step #4: Install CuPy version
    
    pip install cupy-cuda11x
    pip install pynvrtc

# VQVAE (preprocess dataset and train)

    cd VQVAE

    python process_vqvae.py --src ./data/trn --dst ./vqvae_data/trn 
    python process_vqvae.py --src ./data/trn --dst ./vqvae_data/trn --speaker_motion
    
    python process_vqvae.py --src ./data/val --dst ./vqvae_data/val 
    python process_vqvae.py --src ./data/val --dst ./vqvae_data/val --speaker_motion
    
    python train_vqvae.py --trn_folder ./vqvae_data/trn/ --val_folder ./vqvae_data/val/ --serialize_dir ./result/vqvae \
        --force --batch_size 1024

# Featurization

cd ../Featurization

(before running featurization command look into Featurization/Config/FeaturesConfig.py and update all links to yours)

    - in Featurization/Config/FeaturesConfig.py set data_root to 'trn' split and add description as you prefer
    
    python FeatureEmbedding.py  

#TODO: Optimize this with a project dir input in the call of the .py and parametrize everything on it in featuresConfig() 
also to produce both val and trn dataset features at call and not cahnging parameters into FeaturesConfig.py

# Diffusion model train and inference

    cd ../FeaturesDiffuseStyle/mydiffusion_beat_twh
    
    python calculate_gesture_statistics.py --h5_file path/to/your/file.h5

(before next step go into configs/DiffuseStyleGesture.yml and update h5 link to yours and other parameters if needed)
    
    python end2end.py


# Diffusion model inference (generation of new samples)

(from same folder : FeaturesDiffuseStyle/mydiffusion_beat_twh)

    before you have to generate features from your val/test dataset split

    - run featurization changing to data_root 'val' in featurization Config file and run featurization with store_npy

    cd ..local_path/TAG2G/Featurization
    python FeatureEmbedding.py --store_npy

    - alwaiys check to have a config.yml file related to the training in the your model_path to load parameters from 

    cd ../FeaturesDiffuseStyle/mydiffusion_beat_twh
    python sample.py --tst_path=data/val_features --save_dir --model_path 'TWH_mymodel4_512_cinematic/model005379000.pt' --max_length 0 --tst_prefix 'val_2023_v0_023'
    
    Ways of use: 
    1. only tst_path --> data is alredy been featurized (text and audio) as .npy files, load everything from that dir
    2. tst_path & tst_prefix --> load only that sample's features from tst_path directory
    3. if only tst_prefix --> (NOT IMPLEMENTED) script to be built to avoid any kind of previous featurization of the script

Change accordingly both tst_path, model_path, and sample name; be sure to have val dataset featurized using Featurization over both
trn and val dataset (change in the FeaturesConfig.py the parameter dataset_root to val one in your dataset folder)