# Learning to Update Natural Language Comments Based on Code Changes

**Code and datasets for our ACL-2020 paper "Learning to Update Natural Language Comments Based on Code Changes"** which can be found [here](https://arxiv.org/abs/2004.12169).

If you find this work useful, please consider citing our paper:

```
@inproceedings{PanthaplackelETAL20CommentUpdate,
  author = {Panthaplackel, Sheena and Nie, Pengyu and Gligoric, Milos and Li, Junyi Jessy and Mooney, Raymond J.},
  title = {Learning to Update Natural Language Comments Based on Code Changes},
  booktitle = {Association for Computational Linguistics},
  pages = {To appear},
  year = {2020},
}
```

Download generation and update data from [here](https://drive.google.com/open?id=12VMmdE67bp5UFYIoBUf0ibKGXFCH6fQo).

1. Create a directory named `generation-models` in the root directory
```
mkdir generation-models
```
2. Train the comment generation model:
```
python3 comment_generation.py -data_path public_comment_update_data/full_comment_generation/ -model_path generation-models/model.pkl.gz
```
3. Evaluate the comment generation model:
```
python3 comment_generation.py -data_path public_comment_update_data/full_comment_generation/ -model_path generation-models/model.pkl.gz --test_mode
```
4. Create a direction named `embeddings` in the root directory
```
mkdir embeddings
```
5. Save pre-trained embeddings from the comment generation model to disk:
```
python3 comment_generation.py -data_path public_comment_update_data/full_comment_generation/ -model_path generation-models/model.pkl.gz --test_mode --get_embeddings
```
6. Create a directory named `update-models` in the root directory
```
mkdir update-models
```
7. Download `resources` folder from [here](https://drive.google.com/drive/folders/1YZB7FK58LcDCpabj7hlD5vQx_axbdBCQ?usp=sharing), and move it to the root directory.
8. Train the comment update model (i.e., edit model):
```
python3 comment_update.py -data_path public_comment_update_data/comment_update/ -model_path update-models/model.pkl.gz
```
9. Evaluate the comment update model (i.e., edit model):
```
python3 comment_update.py -data_path public_comment_update_data/comment_update/ -model_path update-models/model.pkl.gz --test_mode --rerank
```
