## This file is about how to do preprocessing for several glbs for the trellis pipeline

```bash
conda activate trellis

SRC_DIR=~/Downloads/test_parts
TGT_DIR=/mnt/scratch/trellis_preprocessing

# step 0 get metadata from a folder that you store all the glb files
python dataset_toolkits/build_metadata.py local --input_dir ${SRC_DIR} --output_dir ${TGT_DIR}

# step 1, download/copy the file onto the target folder and also update metadata
python dataset_toolkits/download.py local --output_dir ${TGT_DIR}
python dataset_toolkits/build_metadata.py local --output_dir ${TGT_DIR}

# step 2, render images
python dataset_toolkits/render.py local --output_dir ${TGT_DIR}
python dataset_toolkits/build_metadata.py local --output_dir ${TGT_DIR}

# step 3, voxelize
python dataset_toolkits/voxelize.py local --output_dir ${TGT_DIR}
python dataset_toolkits/build_metadata.py local --output_dir ${TGT_DIR}

# step 4, extract dino feature
python dataset_toolkits/extract_feature.py --output_dir ${TGT_DIR}
python dataset_toolkits/build_metadata.py local --output_dir ${TGT_DIR}

# step 5, encode sparse latent
python dataset_toolkits/encode_ss_latent.py --output_dir ${TGT_DIR}
python dataset_toolkits/build_metadata.py local --output_dir ${TGT_DIR}

# step 6, encoder structure latent
python dataset_toolkits/encode_latent.py --output_dir ${TGT_DIR}
python dataset_toolkits/build_metadata.py local --output_dir ${TGT_DIR}

# step 7, get image condition
python dataset_toolkits/render_cond.py local --output_dir ${TGT_DIR}
python dataset_toolkits/build_metadata.py local --output_dir ${TGT_DIR}
```
