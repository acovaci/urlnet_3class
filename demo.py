from model import URLNet

from loguru import logger

model = URLNet(data_dir = "demo/urls.txt",
               subword_dict_dir = "runs/1000000_emb5_dlm1_32dim_minwf1_1conv3456_5ep_bal3/subwords_dict.p",
               word_dict_dir = "runs/1000000_emb5_dlm1_32dim_minwf1_1conv3456_5ep_bal3/words_dict.p",
               checkpoint_dir = "runs/1000000_emb5_dlm1_32dim_minwf1_1conv3456_5ep_bal3/checkpoints/",
               char_dict_dir = "runs/1000000_emb5_dlm1_32dim_minwf1_1conv3456_5ep_bal3/chars_dict.p",
               emb_mode = 5)

pred, scores = model.predict()

logger.success(pred)
logger.success(scores)