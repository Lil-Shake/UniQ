python train_UniQ_model.py  --resume --num-gpus 4 \
    --config-file configs/UniQ.yaml \
    OUTPUT_DIR checkpoint/UniQ \
    DATASETS.VISUAL_GENOME.IMAGES /home/lxy/datasets/vg/VG_100K \
    DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY /home/lxy/datasets/vg/VG-SGG-dicts-with-attri.json \
    DATASETS.VISUAL_GENOME.IMAGE_DATA /home/lxy/datasets/vg/image_data.json \
    DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 /home/lxy/datasets/vg/VG-SGG-with-attri.h5 \
    MODEL.DETR.OVERSAMPLE_PARAM 0.07 MODEL.DETR.UNDERSAMPLE_PARAM 1.5  SOLVER.CLIP_GRADIENTS.CLIP_VALUE 0.01 \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 160000 \
    SOLVER.STEPS "(130000,)" \
    TEST.EVAL_PERIOD 10000 \
    MODEL.DETR.NO_OBJECT_WEIGHT 0.1 \
    MODEL.DETR.NUM_OBJECT_QUERIES 300 \
    MODEL.DETR.NUM_RELATION_QUERIES 300 \
    MODEL.DETR.COST_CLASS 1.0 \
    MODEL.DETR.GIOU_WEIGHT 1.0 \
    MODEL.DETR.L1_WEIGHT 1.0 \
    MODEL.DETR.GROUP_DETR 3 \
    MODEL.DETR.REWEIGHT_RELATIONS False \
    MODEL.WEIGHTS /home/lxy/lab/IterativeSG/checkpoint/vg_objectdetector_pretrained.pth