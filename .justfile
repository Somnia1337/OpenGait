base-c phase:
  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_da_casiab.yaml --phase {{phase}}

base-o phase:
  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_oumvlp.yaml --phase {{phase}}

set-c phase:
  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 opengait/main.py --cfgs ./configs/gaitset/gaitset.yaml --phase {{phase}}

bin-c phase:
  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 opengait/main.py --cfgs ./configs/bingait/bingait_casiab.yaml --phase {{phase}}

bin-o phase:
  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 opengait/main.py --cfgs ./configs/bingait/bingait_oumvlp.yaml --phase {{phase}}
