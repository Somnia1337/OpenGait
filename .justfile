run := 'CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 opengait/main.py'
null := '/dev/null'

base-c phase log=null:
    {{run}} --cfgs ./configs/gaitbase/gaitbase_da_casiab.yaml --phase {{phase}} 2>&1 | tee {{log}}

base-o phase log=null:
    {{run}} --cfgs ./configs/gaitbase/gaitbase_oumvlp.yaml --phase {{phase}} 2>&1 | tee {{log}}

set-c phase log=null:
    {{run}} --cfgs ./configs/gaitset/gaitset.yaml --phase {{phase}} 2>&1 | tee {{log}}

bin-c phase log=null:
    -{{run}} --cfgs ./configs/bingait/bingait_casiab.yaml --phase {{phase}} 2>&1 | tee {{log}}
    cp {{log}} ./logs/BinGait-CASIAB/{{datetime("%Y%m%d_%H%M")}}.log
    : > {{log}}

bin-o phase log=null:
    -{{run}} --cfgs ./configs/bingait/bingait_oumvlp.yaml --phase {{phase}} 2>&1 | tee {{log}}
    cp {{log}} ./logs/BinGait-OUMVLP/{{datetime("%Y%m%d_%H%M")}}.log
    : > {{log}}
