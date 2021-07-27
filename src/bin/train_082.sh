version='082'
gpu=0

python main_amp_classification2.py --config config/${version}.py --fold 0 --gpu ${gpu}
python main_amp_classification2.py --config config/${version}.py --fold 1 --gpu ${gpu}
python main_amp_classification2.py --config config/${version}.py --fold 2 --gpu ${gpu}
python main_amp_classification2.py --config config/${version}.py --fold 3 --gpu ${gpu}
python main_amp_classification2.py --config config/${version}.py --fold 4 --gpu ${gpu}

python make_oof_submission_cls2.py --version ${version}
