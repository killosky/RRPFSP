
import torch
import random

from env import case_generate, rrpfsp_env
import os
import json


def data_generate():

    with open("./config.json", "r") as f:
        config = json.load(f)
    shop_paras = config["shop_paras"]
    env_paras = config["env_paras"]

    path = os.path.abspath('.') + "/data_dev/"

    shop_info = rrpfsp_env.shop_info_initial(shop_paras)

    case = case_generate.CaseGenerator(shop_info=shop_info, path=path, flag_doc=True)
    for i in range(env_paras["valid_batch_size"]):
        case.get_case(id_instance=i)


if __name__ == "__main__":
    torch.manual_seed(32)
    random.seed(4363)
    data_generate()
