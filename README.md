# Full pipeline commands 

1. Preprocess data and train VQVAE
2. Data featurization (gesture, audio, text)
3. Train diffusion model
4. Inference time diffusion model 

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

    python FeatureEmbedding.py

#TODO: Optimize this with a projecct dir input in the call of the .py and parametrize everything on it in featuresConfig() 
also to produce both val and trn dataset features at call and not cahnging parameters into FeaturesConfig.py

# Diffusion model train and inference

    cd ../FeaturesDiffuseStyle/mydiffusion_beat_twh
    
    python calculate_gesture_statistics.py

(before next step go into configs/DiffuseStyleGesture.yml and update h5 link to yours and other parameters if needed)
    
    python end2end.py


# Diffusion model inferece (generation of new samples)

(from same folder : FeaturesDiffuseStyle/mydiffusion_beat_twh)

    -cinematic version
    python sample.py --config=configs/DiffuseStyleGesture.yml --tst_path=data/val_features --gpu 0 --model_path 'TWH_mymodel4_512_cinematic/model005379000.pt' --max_len 0 --tst_prefix 'val_2023_v0_023'
    
    Ways of use:
    1. only tst_path --> data is alredy been featurized (text and audio) as .npy files, load everything from that dir
    2. tst_path & tst_prefix --> load only that sample's features from tst_path directory
    3. if only tst_prefix --> script to be built to avoid any kind of previous featurization of the script

Change accordingly both tst_path, model_path, and sample name; be sure to have val dataset featurized using Featurization over both
trn and val dataset (change in the FeaturesConfig.py the parameter dataset_root to val one in your dataset folder)