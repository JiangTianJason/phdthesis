# Defend “Training-time” Poisoning attack。Section 6.3.2（2）：Table 6.3、6.4，Figure 6.5；Section 6.3.3（1）：Table 6.10

import os
import time
import torch
import json
from openbackdoor.data import load_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
from openbackdoor.utils import set_config, logger
from openbackdoor.utils.visualize import display_results
from openbackdoor.trainers import load_trainer


def main(config):
    attacker = load_attacker(config["attacker"])
    victim = load_victim(config["victim"])
    target_dataset = load_dataset(**config["target_dataset"])
    poison_dataset = load_dataset(**config["poison_dataset"])
    defender = load_defender(config["defender"])
    defender_type = config['defender']['name']

    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset,defender)
    if config["clean-tune"]:
        logger.info("Fine-tune model on {}".format(config["target_dataset"]["name"]))
        CleanTrainer = load_trainer(config["train"])
        backdoored_model = CleanTrainer.train(backdoored_model, target_dataset)

    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset,defender)

    display = display_results(config, results)

    model_type = config["victim"]['model']
    json_str = json.dumps(display, indent=4)
    with open('./defenders_result/' + defender_type + '-' + model_type + '-t5-encoder-T5CLassificationHead' + '.json', 'a') as json_file:
        json_file.write('\n')
        json_file.write(time.ctime())
        json_file.write('\n')
        json_file.write(json_str)

if __name__ == '__main__':
    for config_path in ['./configs/style_config.json','./configs/syntactic_config.json']:
        with open(config_path, 'r') as f:
            config = json.load(f)

        for model, path in [('bert',r"/root/autodl-tmp/bert-base-uncased")]:

            config["victim"]['model'] = model
            config["victim"]['path'] = path

            config.setdefault('defender', {})
            config['defender']['name'] = 'bki'
            config['defender']["correction"] = True
            config['defender']["pre"] = True
            config['defender']["metrics"] = ["ASR", "CACC"]


            for dataset in ["sst-2","yelp"]:
                config['poison_dataset']['name'] = dataset
                config['target_dataset']['name'] = dataset
                config['attacker']["poisoner"]["poison_rate"] = 0.2
                config['attacker']["poisoner"]["poison_dataset"] = dataset
                config['defender']["poison_dataset"] = dataset
                config['defender']['attacker'] = config['attacker']["poisoner"]['name']
                config['attacker']["train"]["poison_dataset"] = dataset
                config['attacker']["train"]["poison_model"] = model
                config['attacker']['train']["save_path"] = r"./models_after_correction_{}_t5_encoder_T5CLassificationHead".format(config['defender']['name'])
                config['attacker']["poisoner"]["target_label"] = 0
                if dataset in ['yelp', 'sst-2']:
                    config['attacker']["poisoner"]["target_label"] = 1
                config['attacker']['poisoner']['load'] = True
                config['victim']['num_classes'] = 2
                config['defender']['num_classes'] = 2

                if dataset == 'agnews':
                    config['victim']['num_classes'] = 4
                    config['defender']['num_classes'] = 4

                torch.cuda.set_device(0)
                config = set_config(config)
                main(config)
