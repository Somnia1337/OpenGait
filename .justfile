gaitbase phase:
  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_da_casiab.yaml --phase {{phase}}

gaitset phase:
  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 opengait/main.py --cfgs ./configs/gaitset/gaitset.yaml --phase {{phase}}
