import argparse
from stable_baselines3 import PPO
import re
import pandas as pd

# run the following
# python3 extract_parameters.py --model_name model_to_extract_data_from

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='test')
    args = parser.parse_args()

    weight_dir = './csv/'

    model = PPO.load(args.model_name)
    policy = model.policy

    highest_num = 0

    for name, param in policy.named_parameters():
        if param.requires_grad:
            if 'policy' in name and 'net' in name:
                number = int(re.findall(r'\d+', name)[0])
                number = str(int(number/2 + 1))
                if int(number) > highest_num:
                    highest_num = int(number)

                if 'weight' in name:
                    file_name = weight_dir + 'W' + number + '.csv'
                
                elif 'bias' in name:
                    file_name = weight_dir + 'b' + number + '.csv'

                data = param.data.detach().cpu().numpy()
                df = pd.DataFrame(data)
                df.to_csv(file_name, index=False, header=False)
            
            elif 'action' in name:
                if 'weight' in name:
                    file_name = weight_dir + 'W' + str(highest_num + 1) + '.csv'
                
                elif 'bias' in name:
                    file_name = weight_dir + 'b' + str(highest_num + 1) + '.csv'

                data = param.data.detach().cpu().numpy()
                df = pd.DataFrame(data)
                df.to_csv(file_name, index=False, header=False)