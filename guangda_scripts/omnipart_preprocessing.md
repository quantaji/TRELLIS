## This file is about how to do preprocessing for several glbs for the trellis pipeline

```bash
conda activate trellis

SRC_DIR=~/Downloads/pseudo_partverse
PRE_DIR=/mnt/scratch/omnipart_preprocessing
TGT_DIR=/mnt/scratch/omnipart_preprocessed

# step 0 get metadata from a folder that you store all the glb files
python dataset_toolkits/build_metadata.py Partverse --partverse_dir ${SRC_DIR} --output_dir ${PRE_DIR}

# step 1, download/copy the file onto the target folder and also update metadata
python dataset_toolkits/download.py Partverse --output_dir ${PRE_DIR}
python dataset_toolkits/build_metadata.py Partverse --output_dir ${PRE_DIR}

# step 2, render images
python dataset_toolkits/render.py Partverse --output_dir ${PRE_DIR}
python dataset_toolkits/build_metadata.py Partverse --output_dir ${PRE_DIR}

# step 3, voxelize
python dataset_toolkits/voxelize_part.py Partverse --output_dir ${PRE_DIR}
python dataset_toolkits/voxelize_overall.py Partverse --output_dir ${PRE_DIR}
python dataset_toolkits/build_metadata.py Partverse --output_dir ${PRE_DIR}

# step 4, extract dino feature
python dataset_toolkits/extract_feature.py --output_dir ${PRE_DIR}
python dataset_toolkits/build_metadata.py Partverse --output_dir ${PRE_DIR}

# step 4.5, extract sparse feature
python dataset_toolkits/encode_ss_latent.py --output_dir ${PRE_DIR}
python dataset_toolkits/build_metadata.py Partverse --output_dir ${PRE_DIR}


# step 5, encoder structure latent
python dataset_toolkits/encode_latent.py --output_dir ${PRE_DIR}
python dataset_toolkits/build_metadata.py Partverse --output_dir ${PRE_DIR}

# step 6, merge structure latent
python dataset_toolkits/merge_slat.py --preprocess_dir ${PRE_DIR} --omnipart_dir ${TGT_DIR}

# step 7, get part-level image and mask condition
python dataset_toolkits/render_part_img_mask.py --preprocess_dir ${PRE_DIR} --omnipart_dir ${TGT_DIR}

# step 7.5 origional image condition
python dataset_toolkits/render_cond.py Partverse --output_dir ${PRE_DIR}
python dataset_toolkits/build_metadata.py Partverse --output_dir ${PRE_DIR}
```
