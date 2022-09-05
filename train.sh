CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python3 -u train.py -b 512 -w 0,1 -d vgg -n VIT -head CosFace --outdir ./results/ViT-P8S8_ms1m_cosface_s1 --warmup-epochs 1 --lr 3e-4 
