python train_UniQ_model.py --resume --eval-only --num-gpus 4 \
    --config-file configs/UniQ.yaml \
    OUTPUT_DIR checkpoint/UniQ/test \
    DATASETS.VISUAL_GENOME.IMAGES /home/lxy/datasets/vg/VG_100K \
    DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY /home/lxy/datasets/vg/VG-SGG-dicts-with-attri.json \
    DATASETS.VISUAL_GENOME.IMAGE_DATA /home/lxy/datasets/vg/image_data.json \
    DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 /home/lxy/datasets/vg/VG-SGG-with-attri.h5 \
    DATASETS.TEST "('VG_test',)" \
    SOLVER.IMS_PER_BATCH 48 \
    MODEL.WEIGHTS checkpoint/UniQ/model_final.pth