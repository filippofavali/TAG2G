# Full pipeline commands 

1. Preprocess data and train VQVAE
2. Data featurization (gesture, audio, text)
3. Train diffusion model
4. Inference time diffusion model 

# VQVAE (preprocess dataset and train)

cd/VQVAE

python process_vqvae.py --src ./data/trn --dst ./vqvae_data/trn 
python process_vqvae.py --src ./data/trn --dst ./vqvae_data/trn --speaker_motion

python process_vqvae.py --src ./data/val --dst ./vqvae_data/val 
python process_vqvae.py --src ./data/val --dst ./vqvae_data/val --speaker_motion

python train_vqvae.py --trn_folder ./vqvae_data/trn/ --val_folder ./vqvae_data/val/ --serialize_dir ./result/vqvae \
    --force --batch_size 1024

# Featurization

#TODO: insert featurization commands

# Diffusion model train and inference

#TODO: insert diffusion model commands (train and inference)