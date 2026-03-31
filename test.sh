python generate_score.py --gpu 0 --model_path ./ckpt_t1/baseline_specresnet &
python generate_score.py --gpu 1 --model_path ./ckpt_t1/baseline_aasist &
python generate_score.py --gpu 2 --model_path ./ckpt_t1/baseline_ft-w2v2aasist &
python generate_score.py --gpu 3 --model_path ./ckpt_t1/baseline_wpt-w2v2aasist &
python generate_score.py --gpu 4 --model_path ./ckpt_t2/baseline_specresnet &
python generate_score.py --gpu 5 --model_path ./ckpt_t2/baseline_aasist &
python generate_score.py --gpu 6 --model_path ./ckpt_t2/baseline_ft-w2v2aasist &
python generate_score.py --gpu 7 --model_path ./ckpt_t2/baseline_wpt-w2v2aasist 