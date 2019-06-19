from deeppavlov.deep import find_config
from deeppavlov.core.commands.infer import build_model

config_path = find_config('elmo_lm_ready4fine_tuning_ru_news')
model = build_model(config_path, load_trained = True, download = True)
print(model)

