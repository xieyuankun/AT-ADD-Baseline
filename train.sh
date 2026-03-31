python main_train.py --gpu 0 --train_task atadd-track1 --model specresnet --o ./ckpt_t1/baseline_specresnet &
python main_train.py --gpu 1 --train_task atadd-track1 --model aasist --lr 0.0001 --batch_size 24 --o ./ckpt_t1/baseline_aasist &
python main_train.py --gpu 2 --train_task atadd-track1 --model ft-w2v2aasist --seed 1234 --batch_size 14 --lr 0.000001 --o ./ckpt_t1/baseline_ft-w2v2aasist &
python main_train.py --gpu 3 --train_task atadd-track1 --model wpt-w2v2aasist --batch_size 32 --num_prompt_tokens 6 --num_wavelet_tokens 4 --o ./ckpt_t1/baseline_wpt-w2v2aasist &
python main_train.py --gpu 4 --train_task atadd-track2 --model specresnet --num_epochs 10 --interval 2 --o ./ckpt_t2/baseline_specresnet &
python main_train.py --gpu 5 --train_task atadd-track2 --model aasist --num_epochs 10 --interval 2 --lr 0.0001 --batch_size 24 --o ./ckpt_t2/baseline_aasist &
python main_train.py --gpu 6 --train_task atadd-track2 --model ft-w2v2aasist --num_epochs 10 --interval 2 --seed 1234 --batch_size 14 --lr 0.000001 --o ./ckpt_t2/baseline_ft-w2v2aasist &
python main_train.py --gpu 7 --train_task atadd-track2 --model wpt-w2v2aasist --num_epochs 10 --interval 2 --batch_size 32 --num_prompt_tokens 6 --num_wavelet_tokens 4 --o ./ckpt_t2/baseline_wpt-w2v2aasist 
